# -*- coding: utf-8 -*-
"""
Created on Thu May 15 21:57:38 2025

@author: User
"""

# -*- coding: utf-8 -*-
"""
4-Genre CNN 모델 (Dynamic Class Support)
- 원하는 장르만 TARGET_GENRES 리스트에 추가하여 분류 가능
- TARGET_GENRES가 빈 리스트([])일 경우 CSV에 정의된 모든 장르 사용
- num_classes는 label_map 길이에서 자동 계산
- history 저장 방식은 원본 코드와 동일하게 에포크별, 테스트 결과 저장
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from preprocess.preprocess import dataToDict

# ----------------------------
# 사용자 설정: 분류할 장르 목록
# ----------------------------
# 빈 리스트([])로 두면 CSV에 정의된 모든 장르 사용
TARGET_GENRES = ['Rock', 'Electronic', 'Experimental', 'Hip-Hop']

# ----------------------------
# 1) Dataset 정의
# ----------------------------
class LazyHyperImageDataset(Dataset):
    """
    .npz 하이퍼이미지를 메모리맵 모드로 로드하는 Dataset
    """
    def __init__(self, paths, labels):
        self.paths  = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 'hyper' 키로 저장된 배열을 float32로 로드하고 tensor로 변환
        arr = np.load(self.paths[idx], mmap_mode='r')['hyper'].astype(np.float32)
        x   = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        y   = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ----------------------------
# 2) 모델 정의 (HyperCNN)
# ----------------------------
class HyperCNN(nn.Module):
    """
    전역 풀링 기반 CNN
    """
    def __init__(self, num_classes):
        super().__init__()
        # 특징 추출부
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.LocalResponseNorm(5), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.LocalResponseNorm(5), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.ReLU(), nn.LocalResponseNorm(5), nn.MaxPool2d(2),
        )
        # 전역 평균 풀링 → (batch, 128,1,1)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        # 분류기
        self.classifier = nn.Sequential(
            nn.Flatten(),                   # → (batch,128)
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),   # num_classes 자동 적용
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)

# ----------------------------
# 3) Top-k Accuracy
# ----------------------------
def topk_accuracy(output, target, k=1):
    _, pred = output.topk(k, dim=1)
    correct_k = pred.eq(target.view(-1,1)).any(1).float().sum().item()
    return correct_k / target.size(0)

# ----------------------------
# 4) 메인 함수
# ----------------------------
def main():
    # (1) 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = (device.type=='cuda')

    # (2) 데이터 로드
    """
    # 16 장르 train/val/test split csv
    CSV_PATH = './preprocess/track_filtered.csv'
    # 처음 버전(628차원) 하이퍼이미지
    NPZ_DIR  = 'hyperimage'
    """
    # 4 장르 train/val/test split csv
    CSV_PATH = './preprocess/track_small.csv'
    # 개선된 버전(512 차원) 하이퍼이미지
    NPZ_DIR  = 'hyperimage_v3'
    tracks   = dataToDict(CSV_PATH, NPZ_DIR)

    # (3) TARGET_GENRES 필터링
    if TARGET_GENRES:
        tracks = {tid:info for tid,info in tracks.items()
                  if info['genre_top'] in TARGET_GENRES}

    # (4) 경로 & 장르 리스트 생성
    tids       = list(tracks.keys())
    paths_all  = [f"{NPZ_DIR}/{tid}.npz" for tid in tids]
    genres_all = [tracks[tid]['genre_top'] for tid in tids]

    # label_map 동적 생성: 장르명→정수
    label_map  = {g:i for i,g in enumerate(sorted(set(genres_all)))}
    labels_all = [label_map[g] for g in genres_all]

    # (5) split에 따라 Train/Val/Test 분리
    train_paths  = [p for p,t in zip(paths_all, tids) if tracks[t]['split']=='training']
    train_labels = [labels_all[i] for i,t in enumerate(tids) if tracks[t]['split']=='training']
    val_paths    = [p for p,t in zip(paths_all, tids) if tracks[t]['split']=='validation']
    val_labels   = [labels_all[i] for i,t in enumerate(tids) if tracks[t]['split']=='validation']
    test_paths   = [p for p,t in zip(paths_all, tids) if tracks[t]['split']=='test']
    test_labels  = [labels_all[i] for i,t in enumerate(tids) if tracks[t]['split']=='test']

    # (6) label_map 저장 (선택)
    with open('label_map_dynamic.json','w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # (7) DataLoader 설정
    BATCH_SIZE = 16
    pin_memory = (device.type=='cuda')
    train_loader = DataLoader(LazyHyperImageDataset(train_paths, train_labels),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=pin_memory)
    val_loader   = DataLoader(LazyHyperImageDataset(val_paths,   val_labels),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=pin_memory)
    test_loader  = DataLoader(LazyHyperImageDataset(test_paths,  test_labels),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=pin_memory)

    # (8) 모델/손실/최적화/스케줄러/스케일러 설정
    num_classes = len(label_map)  # TARGET_GENRES 길이에 따라 자동 계산
    model       = HyperCNN(num_classes).to(device)
    criterion   = nn.CrossEntropyLoss()
    optimizer   = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=0.5, patience=3)
    scaler      = torch.amp.GradScaler()

    # (9) 학습 루프 및 history 저장
    best_f1, early_stop = 0, 0
    NUM_EPOCHS = 30
    history = {}
    for epoch in tqdm(range(1, NUM_EPOCHS+1), desc="Epochs", unit="ep"):
        # -- Train
        model.train()
        train_loss = 0.0
        for x,y in tqdm(train_loader, desc=f"Train {epoch}/{NUM_EPOCHS}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out  = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            train_loss += loss.item()*x.size(0)
        train_loss /= len(train_loader.dataset)

        # -- Validate
        model.eval()
        t1, t3 = 0.0, 0.0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for x,y in tqdm(val_loader, desc=f"Val {epoch}/{NUM_EPOCHS}", leave=False):
                x, y = x.to(device), y.to(device)
                out = model(x)
                t1 += topk_accuracy(out, y, 1)*x.size(0)
                t3 += topk_accuracy(out, y, 3)*x.size(0)
                val_preds.extend(out.argmax(1).cpu().tolist())
                val_trues.extend(y.cpu().tolist())
        val_top1 = t1/len(val_loader.dataset)
        val_top3 = t3/len(val_loader.dataset)
        val_f1   = f1_score(val_trues, val_preds, average='macro')
        # 에포크별 성능 기록
        history[epoch] = {'train_loss':train_loss,
                          'val_top1':val_top1,'val_top3':val_top3,'val_f1':val_f1}
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_top1={val_top1:.3f},"
              f" val_top3={val_top3:.3f}, val_f1={val_f1:.3f}")
        # Early stopping & 모델 저장
        scheduler.step(val_f1)
        if val_f1 > best_f1:
            best_f1, early_stop = val_f1, 0
            torch.save(model.state_dict(), 'result/best_hypercnn_4genres.pth')
        else:
            early_stop += 1
            if early_stop >= 5:
                print("Early stopping triggered")
                break

    # (10) Test 평가
    model.load_state_dict(torch.load('result/best_hypercnn_4genres.pth', map_location=device))
    model.eval()
    t1, t3 = 0.0, 0.0
    test_preds, test_trues = [], []
    with torch.no_grad():
        for x,y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            t1 += topk_accuracy(out, y, 1)*x.size(0)
            t3 += topk_accuracy(out, y, 3)*x.size(0)
            test_preds.extend(out.argmax(1).cpu().tolist())
            test_trues.extend(y.cpu().tolist())
    test_top1 = t1/len(test_loader.dataset)
    test_top3 = t3/len(test_loader.dataset)
    test_f1   = f1_score(test_trues, test_preds, average='macro')
    test_cm   = confusion_matrix(test_trues, test_preds)
    # 테스트 결과 저장
    history['test'] = {'test_top1':test_top1, 'test_top3':test_top3,
                       'test_f1':test_f1, 'test_cm':test_cm.tolist()}
    with open('result/hypercnn_results_2.json', 'w') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 최종 출력
    print(f"Test Top-1: {test_top1:.3f}, Top-3: {test_top3:.3f}")

if __name__ == '__main__':
    main()
