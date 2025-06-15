# -*- coding: utf-8 -*-
"""
Created on Mon May 19 00:46:58 2025

@author: User
"""

"""
5-Fold 교차검증 기반 16장르 분류 HyperCNN
- StratifiedKFold(n_splits=5)로 Fold별 Train/Val 분할
- Class-Weighted Loss를 사용하지 않고 Base HyperCNN 구조 유지
- Metrics: train_loss, P@1(top1 acc), P@3(top3 acc), F1 기록 및 Fold별 best 모델 저장
- AMP, Early Stopping, ReduceLROnPlateau 스케줄러 적용
- 에포크별 성능 기록(history)에 저장 후 JSON 파일로 출력
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
# 1) Dataset 정의: LazyHyperImageDataset
# ----------------------------
class LazyHyperImageDataset(Dataset):
    """
    .npz 파일에 저장된 하이퍼이미지를 메모리맵 모드(r)로 로드
    반환값: (Tensor[1, H, W], label)
    """
    def __init__(self, paths, labels):
        self.paths = paths      # 하이퍼이미지 파일 경로 리스트
        self.labels = labels    # 정수형 레이블 리스트

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 하이퍼이미지('hyper' 키) 로드 후 float32 변환
        arr = np.load(self.paths[idx], mmap_mode='r')['hyper'].astype(np.float32)
        x = torch.from_numpy(arr).unsqueeze(0)  # 차원 확장: (1, H, W)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ----------------------------
# 2) 모델 정의: HyperCNN
# ----------------------------
class HyperCNN(nn.Module):
    """
    Base CNN 구조:
      - 3×(Conv2d→ReLU→LocalResponseNorm→MaxPool2d)
      - 전역 평균 풀링(GlobalAvgPool)
      - Flatten → FC(128→256) → ReLU → Dropout(0.3) → FC(256→num_classes)
    """
    def __init__(self, num_classes, height=512, width=1287):
        super().__init__()
        # 특징 추출부: 합성곱+정규화+풀링 블록
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.LocalResponseNorm(5), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.LocalResponseNorm(5), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3, padding=1), nn.ReLU(), nn.LocalResponseNorm(5), nn.MaxPool2d(2),
        )
        # 전역 평균 풀링: (batch, 128, 1, 1)로 고정
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        # 분류기: 128차원 → 256 → num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)      # (batch,128,H',W')
        x = self.global_pool(x)   # (batch,128,1,1)
        return self.classifier(x) # (batch,num_classes)

# ----------------------------
# 3) Top-k 정확도 계산 함수
# ----------------------------
def topk_accuracy(output, target, k=1):
    """
    output: 모델 출력 로짓, shape=(batch, num_classes)
    target: 정답 레이블, shape=(batch,)
    k: Top-k
    return: Top-k 정확도 (0~1 사이 값)
    """
    # 상위 k개 예측 인덱스 추출 → (batch, k)
    _, pred = output.topk(k, dim=1)
    # 실제 label이 상위 k개에 포함된 경우 count
    correct_k = pred.eq(target.view(-1,1)).any(1).float().sum().item()
    return correct_k / target.size(0)

# ----------------------------
# 4) Main 함수: 5-Fold 교차검증
# ----------------------------
def main():
    # (1) 디바이스 설정 및 AMP 스케일러
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = (device.type=='cuda')
    scaler = torch.amp.GradScaler()

    # (2) 데이터 로드: track_filtered.csv → tracks dict
    CSV_PATH = './preprocess/track_filtered.csv'
    NPZ_DIR = 'hyperimage_16genres'
    tracks = dataToDict(CSV_PATH, NPZ_DIR)
    with open('label_map_16genres.json','r') as f:
        label_map = json.load(f)  # 장르명→정수 매핑

    # 전체 샘플 경로 및 레이블 리스트 생성
    tids = list(tracks.keys())
    paths_all = [f"{NPZ_DIR}/{tid}.npz" for tid in tids]
    labels_all = [label_map[tracks[tid]['genre_top']] for tid in tids]

    # 글로벌 혼돈행렬, 히스토리 저장소
    all_trues_global, all_preds_global = [], []
    all_history = {}

    # (3) StratifiedKFold로 5개 Fold 분할
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(paths_all, labels_all), 1):
        # a) Fold별 train/val 데이터 준비
        train_paths  = [paths_all[i] for i in train_idx]
        train_labels = [labels_all[i] for i in train_idx]
        val_paths    = [paths_all[i] for i in val_idx]
        val_labels   = [labels_all[i] for i in val_idx]

        # b) DataLoader 생성
        train_loader = DataLoader(LazyHyperImageDataset(train_paths, train_labels),
                                  batch_size=16, shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(LazyHyperImageDataset(val_paths,   val_labels),
                                  batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

        # (4) 모델/손실/최적화/스케줄러 설정
        num_classes = len(label_map)
        model       = HyperCNN(num_classes).to(device)
        criterion   = nn.CrossEntropyLoss()
        optimizer   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler   = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        # (5) Epoch 루프: 학습 및 검증, Early Stopping
        best_f1, no_improve = 0, 0
        history = {}
        for epoch in tqdm(range(1, 31), desc=f"Fold{fold_idx} Epochs", unit="ep"):
            # -- 학습 단계
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
                train_loss += loss.item() * x.size(0)
            train_loss /= len(train_loader.dataset)

            # -- 검증 단계
            model.eval()
            t1_sum, t3_sum = 0.0, 0.0
            val_preds, val_trues = [], []
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc="  Val", leave=False):
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    t1_sum += topk_accuracy(out, y, 1) * x.size(0)
                    t3_sum += topk_accuracy(out, y, 3) * x.size(0)
                    preds = out.argmax(1).cpu().tolist()
                    val_preds.extend(preds); val_trues.extend(y.cpu().tolist())

            # 글로벌 혼돈행렬 업데이트
            all_trues_global.extend(val_trues)
            all_preds_global.extend(val_preds)

            # 메트릭 계산 및 기록
            val_top1 = t1_sum / len(val_loader.dataset)
            val_top3 = t3_sum / len(val_loader.dataset)
            val_f1   = f1_score(val_trues, val_preds, average='macro')
            history[epoch] = {'train_loss': train_loss,
                              'P@1': val_top1, 'acc1': val_top1,
                              'P@3': val_top3, 'acc3': val_top3,
                              'F1': val_f1}
            print(f"Fold{fold_idx} E{epoch}: loss={train_loss:.4f}, P@1={val_top1:.3f}, P@3={val_top3:.3f}, F1={val_f1:.3f}")

            # 학습률 스케줄러 및 Early Stopping
            scheduler.step(val_f1)
            if val_f1 > best_f1:
                best_f1, no_improve = val_f1, 0
                torch.save(model.state_dict(), f'result/best_model_fold{fold_idx}.pth')
            else:
                no_improve += 1
                if no_improve >= 5:
                    print(f"Fold{fold_idx} 조기 종료: epoch {epoch}")
                    break

        # Fold별 히스토리 저장
        all_history[f'fold{fold_idx}'] = history

    # (6) 히스토리 전체 JSON 저장
    with open('result/hypercnn_5fold_results.json', 'w') as f:
        json.dump(all_history, f, ensure_ascii=False, indent=2)

    # (7) 전체 Fold 혼돈행렬 생성 및 저장
    cm_total = confusion_matrix(all_trues_global, all_preds_global)
    print('전체 5-Fold 혼돈행렬:')
    print(cm_total)
    with open('result/hypercnn_5fold_overall_cm.json', 'w') as f:
        json.dump({'overall_confusion_matrix': cm_total.tolist()}, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
