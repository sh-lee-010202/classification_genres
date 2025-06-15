# -*- coding: utf-8 -*-
"""
Created on Sat May 24 14:24:39 2025

@author: User
"""

# -*- coding: utf-8 -*-
"""
5-Fold 교차검증 기반 16장르 분류 (Class-Weighted Loss 적용 버전)
- TARGET_GENRES 리스트로 분류할 장르를 동적으로 설정 가능
  * 빈 리스트([])일 경우 CSV에 정의된 모든 장르 사용
- StratifiedKFold(n_splits=5)로 Fold별 Train/Val 분할
- Class-Weighted CrossEntropyLoss로 소수 클래스 손실 가중치 보정
- AMP, Early Stopping, ReduceLROnPlateau 스케줄러 적용
- 에포크별, Fold별 성능(history)에 기록 후 JSON 저장
- 전체 Fold의 혼돈행렬(overall confusion matrix) 저장
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from preprocess.preprocess import dataToDict

# ----------------------------
# 사용자 설정: 분류에 사용할 장르 목록
# ----------------------------
# 빈 리스트([])로 두면 CSV에 정의된 모든 장르 사용
TARGET_GENRES = []

# ----------------------------
# 1) Dataset 정의: LazyHyperImageDataset
# ----------------------------
class LazyHyperImageDataset(Dataset):
    """
    npz로 저장된 하이퍼이미지를 메모리맵 모드로 로드하여
    (Tensor[1,H,W], 레이블) 형태로 반환
    """
    def __init__(self, paths, labels):
        self.paths  = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = np.load(self.paths[idx], mmap_mode='r')['hyper'].astype(np.float32)
        x   = torch.from_numpy(arr).unsqueeze(0)
        y   = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ----------------------------
# 2) 모델 정의: HyperCNN 클래스
# ----------------------------
class HyperCNN(nn.Module):
    """
    3회 Conv+ReLU+LRN+MaxPool → GlobalAvgPool → FC 구조
    """
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.LocalResponseNorm(5), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.LocalResponseNorm(5), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3, padding=1), nn.ReLU(), nn.LocalResponseNorm(5), nn.MaxPool2d(2),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128,256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)

# ----------------------------
# 3) Top-k 정확도 함수 정의
# ----------------------------
def topk_accuracy(output, target, k=1):
    _, pred = output.topk(k, dim=1)
    correct_k = pred.eq(target.view(-1,1)).any(1).float().sum().item()
    return correct_k / target.size(0)

# ----------------------------
# 4) Main 함수: 5-Fold CV + Class-Weighted Loss + TARGET_GENRES 적용
# ----------------------------
def main():
    # (1) 장치 및 AMP 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = (device.type=='cuda')
    scaler = torch.amp.GradScaler()

    # (2) CSV→tracks 로드, label_map 불러오기
    CSV_PATH = './preprocess/track_filtered.csv'
    NPZ_DIR  = 'hyperimage_16genres'
    tracks   = dataToDict(CSV_PATH, NPZ_DIR)
    with open('label_map_16genres.json','r') as f:
        base_label_map = json.load(f)  # 전체 장르명→정수

    # (3) TARGET_GENRES 필터링: 리스트에 있거나 전체 사용
    if TARGET_GENRES:
        # tracks에서 genre_top이 TARGET_GENRES에 포함된 항목만 선별
        tracks = {tid:info for tid,info in tracks.items() if info['genre_top'] in TARGET_GENRES}
        # label_map도 TARGET_GENRES 순서대로 재구성
        label_map = {g:i for i,g in enumerate(TARGET_GENRES)}
    else:
        label_map = base_label_map

    # (4) 전체 트랙 ID, 경로, 레이블 리스트 생성
    tids       = list(tracks.keys())
    paths_all  = [f"{NPZ_DIR}/{tid}.npz" for tid in tids]
    labels_all = [label_map[tracks[tid]['genre_top']] for tid in tids]

    # (5) Fold별 전체 혼돈행렬·history 저장 변수
    all_trues_global, all_preds_global = [], []
    all_history = {}

    # (6) 5-Fold 교차검증을 위한 StratifiedKFold 설정
    #     - 전체 데이터를 균등한 클래스 분포를 유지하도록 5개의 Fold로 분할
    #     - 각 Fold는 한 번씩 검증(Validation) 세트 역할을 하고, 나머지 4개 Fold는 학습(Train) 세트가 됨
    #     - 이렇게 5번 반복하여 모델의 일반화 성능을 안정적으로 평가
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # (7) Fold 순회: 각 fold마다 별도의 모델 학습 및 검증 수행
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(paths_all, labels_all), 1):
        # (6a) Fold별 Train/Val 분할
        train_paths  = [paths_all[i] for i in train_idx]
        train_labels = [labels_all[i] for i in train_idx]
        val_paths    = [paths_all[i] for i in val_idx]
        val_labels   = [labels_all[i] for i in val_idx]

        # (6b) DataLoader 생성
        train_loader = DataLoader(LazyHyperImageDataset(train_paths,  train_labels), batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(LazyHyperImageDataset(val_paths,    val_labels),   batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

        # (7) 모델 및 손실함수, 옵티마이저, 스케줄러 설정
        num_classes = len(label_map)
        model       = HyperCNN(num_classes).to(device)
        # -- 클래스별 샘플 수 집계 및 역빈도 기반 가중치 계산
        counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
        class_weights = torch.tensor(counts.sum()/(num_classes*counts), dtype=torch.float32).to(device)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)
        optimizer     = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler     = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        # (8) 학습 루프: Epoch별 학습·검증·모델 저장·Early Stopping
        best_f1, no_improve = 0, 0
        history = {}
        for epoch in tqdm(range(1,31), desc=f"Fold{fold_idx} Epochs", unit="ep"):
            # --- Train 단계
            model.train()
            train_loss = 0.0
            for x, y in tqdm(train_loader, desc="  Train", leave=False):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    out  = model(x)
                    loss = criterion(out, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                train_loss += loss.item()*x.size(0)
            train_loss /= len(train_loader.dataset)

            # --- Validation 단계
            model.eval()
            t1_sum, t3_sum = 0.0, 0.0
            val_preds, val_trues = [], []
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc="  Val", leave=False):
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    t1_sum += topk_accuracy(out,y,1) * x.size(0)
                    t3_sum += topk_accuracy(out,y,3) * x.size(0)
                    preds = out.argmax(1).cpu().tolist()
                    val_preds.extend(preds); val_trues.extend(y.cpu().tolist())

            # Fold별 global confusion update
            all_trues_global.extend(val_trues)
            all_preds_global.extend(val_preds)

            # Metric 계산 및 기록
            val_top1 = t1_sum / len(val_loader.dataset)
            val_top3 = t3_sum / len(val_loader.dataset)
            val_f1   = f1_score(val_trues, val_preds, average='macro')
            history[epoch] = {'train_loss':train_loss, 'P@1':val_top1, 'acc1':val_top1,
                              'P@3':val_top3, 'acc3':val_top3, 'F1':val_f1}
            print(f"Fold{fold_idx} E{epoch}: loss={train_loss:.4f}, P@1={val_top1:.3f}, P@3={val_top3:.3f}, F1={val_f1:.3f}")

            # Scheduler & Early Stopping & 모델 저장
            scheduler.step(val_f1)
            if val_f1 > best_f1:
                best_f1, no_improve = val_f1, 0
                torch.save(model.state_dict(), f'result/best_model_fold{fold_idx}_w.pth')
            else:
                no_improve += 1
                if no_improve >= 5:
                    print(f"Fold{fold_idx} 조기 종료 at epoch {epoch}")
                    break

        all_history[f'fold{fold_idx}'] = history

    # (9) 전체 Fold 결과 JSON 저장
    with open('result/hypercnn_5fold_results_w.json','w') as f:
        json.dump(all_history, f, ensure_ascii=False, indent=2)

    # (10) 전체 혼돈행렬 계산 및 저장
    cm_total = confusion_matrix(all_trues_global, all_preds_global)
    print('전체 5-Fold 혼돈행렬:')
    print(cm_total)
    with open('result/hypercnn_5fold_overall_cm_w.json','w') as f:
        json.dump({'overall_confusion_matrix': cm_total.tolist()}, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()



