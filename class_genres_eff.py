# -*- coding: utf-8 -*-
"""
Created on Wed May 21 20:18:44 2025

@author: User
"""

# -*- coding: utf-8 -*-
"""
장르 분류 EfficientNet-B0 (동적 클래스 지원, Fold 및 Class-Weight 미적용)
- TARGET_GENRES: 분류할 장르 리스트 지정 (빈 리스트([])일 경우 CSV에 정의된 모든 장르 사용)
- Train/Val/Test 분리: preprocessed CSV의 'split' 필드 기반
- Class-Weighted Loss, Fold CV 미적용: 단일 분할로 학습/검증/테스트 수행
- AMP, ReduceLROnPlateau 스케줄러, Early Stopping 적용
- 에포크별 및 최종 테스트 성능을 history에 기록하고 JSON 저장
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0
from preprocess.preprocess import dataToDict

# ----------------------------
# 사용자 설정: 분류할 장르 목록
# ----------------------------
# 빈 리스트([])로 두면 CSV에 있는 모든 장르 사용
TARGET_GENRES = []

# ----------------------------
# Dataset 클래스 정의
# ----------------------------
class LazyHyperImageDataset(Dataset):
    """
    하이퍼이미지(.npz) 파일을 메모리맵으로 로드하여
    모델 입력용 tensor로 반환하는 Dataset
    """
    def __init__(self, paths, labels):
        self.paths = paths
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        arr = np.load(self.paths[idx], mmap_mode='r')['hyper'].astype(np.float32)
        x = torch.from_numpy(arr).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ----------------------------
# 모델 정의: EfficientNet-B0 기반
# ----------------------------
class HyperEfficientNet(nn.Module):
    """
    EfficientNet-B0 백본 + 입력채널 변경 + 출력 클래스 조정
    """
    def __init__(self, num_classes):
        super().__init__()
        base = efficientnet_b0(weights=None)
        # 입력 채널을 1로 변경
        base.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # 최종 분류기 조정
        in_feats = base.classifier[1].in_features
        base.classifier[1] = nn.Linear(in_feats, num_classes)
        self.net = base

    def forward(self, x):
        return self.net(x)

# ----------------------------
# 평가 지표: Top-k Accuracy
# ----------------------------
def topk_accuracy(output, target, k=1):
    _, pred = output.topk(k, dim=1)
    correct = pred.eq(target.view(-1,1)).any(1).float().mean().item()
    return correct

# ----------------------------
# 메인 함수: 학습/검증/테스트 수행
# ----------------------------
def main():
    # 1) 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = (device.type == 'cuda')

    # 2) 데이터 로드
    CSV_PATH = './preprocess/track_filtered.csv'
    NPZ_DIR = 'hyperimage_16genres'
    tracks = dataToDict(CSV_PATH, NPZ_DIR)

    # 3) TARGET_GENRES 필터링
    if TARGET_GENRES:
        tracks = {tid:info for tid, info in tracks.items() if info['genre_top'] in TARGET_GENRES}

    # 4) 라벨 매핑 생성
    genres = sorted({info['genre_top'] for info in tracks.values()})
    label_map = {g:i for i,g in enumerate(genres)}
    num_classes = len(label_map)
    with open('label_map_dynamic.json','w',encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # 5) 경로 및 라벨 리스트
    tids = list(tracks.keys())
    paths_all = [f'{NPZ_DIR}/{tid}.npz' for tid in tids]
    #labels_all = [label_map[tracks[tid]['genre_top']] for tid in tids]

    # 6) Train/Val/Test 분할
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    for path, tid in zip(paths_all, tids):
        split = tracks[tid]['split']
        lbl = label_map[tracks[tid]['genre_top']]
        if split == 'training':
            train_paths.append(path); train_labels.append(lbl)
        elif split == 'validation':
            val_paths.append(path); val_labels.append(lbl)
        else:
            test_paths.append(path); test_labels.append(lbl)

    # 7) DataLoader 설정
    BATCH_SIZE = 32
    pin_mem = (device.type == 'cuda')
    train_loader = DataLoader(LazyHyperImageDataset(train_paths, train_labels), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=pin_mem)
    val_loader = DataLoader(LazyHyperImageDataset(val_paths, val_labels), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=pin_mem)
    test_loader = DataLoader(LazyHyperImageDataset(test_paths, test_labels), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=pin_mem)

    # 8) 모델 및 최적화 설정
    model = HyperEfficientNet(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler()

    # 9) 학습/검증 루프
    best_f1 = 0.0
    early_stop = 0
    history = {}
    EPOCHS = 30
    for epoch in range(1, EPOCHS+1):
        # 학습 단계
        model.train()
        total_loss = 0.0
        for x_cpu, y_cpu in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} Train", leave=False):
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        # 검증 단계
        model.eval()
        t1_sum, t3_sum = 0.0, 0.0
        preds, tgts = [], []
        with torch.no_grad():
            for x_cpu, y_cpu in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} Val", leave=False):
                x = x_cpu.to(device); y = y_cpu.to(device)
                out = model(x)
                # Top-1, Top-3 정확도 누적 (맞춘 샘플 수)
                t1_sum += topk_accuracy(out, y, k=1) * x.size(0)
                t3_sum += topk_accuracy(out, y, k=3) * x.size(0)
                preds.extend(out.argmax(1).cpu().numpy())
                tgts.extend(y.cpu().numpy())
        val_top1 = t1_sum / len(val_loader.dataset)
        val_top3 = t3_sum / len(val_loader.dataset)
        val_f1 = f1_score(tgts, preds, average='macro')
        val_cm = confusion_matrix(tgts, preds).tolist()

        # 스케줄러 업데이트 & Early Stopping
        scheduler.step(val_f1)
        history[epoch] = {'train_loss':train_loss,'val_top1':val_top1,'val_top3':val_top3,'val_f1':val_f1,'val_cm':val_cm}
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_top1={val_top1:.3f}, val_top3={val_top3:.3f}, val_f1={val_f1:.3f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            early_stop = 0
            torch.save(model.state_dict(), 'result/best_efficientnet_b0_16.pth')
        else:
            early_stop += 1
            if early_stop >= 5:
                print('>> Early stopping')
                break

    # 10) 최종 테스트 평가
    model.load_state_dict(torch.load('result/best_efficientnet_b0_16.pth', map_location=device))
    model.eval()
    t1_sum, t3_sum = 0.0, 0.0
    preds, tgts = [], []
    with torch.no_grad():
        for x_cpu, y_cpu in tqdm(test_loader, desc='[Test]'): 
            x = x_cpu.to(device); y = y_cpu.to(device)
            out = model(x)
            t1_sum += topk_accuracy(out, y, k=1) * x.size(0)
            t3_sum += topk_accuracy(out, y, k=3) * x.size(0)
            preds.extend(out.argmax(1).cpu().numpy())
            tgts.extend(y.cpu().numpy())
    test_top1 = t1_sum / len(test_loader.dataset)
    test_top3 = t3_sum / len(test_loader.dataset)
    test_f1 = f1_score(tgts, preds, average='macro')
    test_cm = confusion_matrix(tgts, preds).tolist()
    history['test'] = {'test_top1':test_top1,'test_top3':test_top3,'test_f1':test_f1,'test_cm':test_cm}

    # 결과 저장 및 출력
    with open('result/efficientnet_b0_16_results.json','w',encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[Test] top1={test_top1:.3f}, top3={test_top3:.3f}, f1={test_f1:.3f}")

if __name__ == '__main__':
    main()




