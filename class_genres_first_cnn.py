# -*- coding: utf-8 -*-
"""
Created on Thu May 15 21:57:38 2025

@author: User
"""
#set OMP_NUM_THREADS=1

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from preprocess.preprocess import dataToDict


# ─────────────────────────────────────────────────────────
# 1) Dataset
# ─────────────────────────────────────────────────────────
class LazyHyperImageDataset(Dataset):
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
# 2) 모델 정의
# ----------------------------
"""
# 1차 학습
class HyperCNN(nn.Module):
    def __init__(self, num_classes, height=512, width=1287):
        super().__init__()
        # 3-layer CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.LocalResponseNorm(5),                    nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.LocalResponseNorm(5),                     nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.LocalResponseNorm(5),                      nn.MaxPool2d(2),
        )

        # 더미로 feature map 크기 계산
        with torch.no_grad():
            dummy = torch.zeros(1, 1, height, width, device=next(self.features.parameters()).device)
            feat  = self.features(dummy)
            n_feats = feat.numel()  # = feat.shape[1]*feat.shape[2]*feat.shape[3]

        # classifier: n_feats 에 맞춰 자동으로 생성
        self.classifier = nn.Sequential(
            nn.Linear(n_feats, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
"""
# 2차 학습 (전역 풀링 > FC 레이어 차원 대폭 감소)
class HyperCNN(nn.Module):
    def __init__(self, num_classes, height=512, width=1287):
        super().__init__()
        # 3-layer CNN feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.LocalResponseNorm(5),        nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.LocalResponseNorm(5),         nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.LocalResponseNorm(5),          nn.MaxPool2d(2),
        )
        # 전역 평균 풀링
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))  # → (batch, 128, 1, 1)
        # classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),                # → (batch, 128)
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)           # → (batch, 128, H’, W’)
        x = self.global_pool(x)        # → (batch, 128, 1, 1)
        return self.classifier(x)      # → (batch, num_classes)


# ----------------------------
# 3) 지표 함수
# ----------------------------
def topk_accuracy(output, target, k=1):
    _, pred = output.topk(k, dim=1)
    return pred.eq(target.view(-1,1)).any(1).float().mean().item()

# ----------------------------
# 4) main 함수
# ----------------------------
def main():
    # (1) 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = (device.type=='cuda')

    # (2) 데이터 로드
    CSV_PATH = './preprocess/track_filtered.csv'
    #NPZ_DIR  = 'arrays'
    NPZ_DIR = 'hyperimage_v2'
    tracks   = dataToDict(CSV_PATH, NPZ_DIR)
    
    # 4개 장르만 남기기
    target_genres = ['Rock', 'Electronic', 'Experimental', 'Hip-Hop']
    
    tracks = {
        tid:info for tid,info in tracks.items()
        if info['genre_top'] in target_genres
    }

    # (3) 경로, 레이블 리스트 생성
    tids       = list(tracks.keys())
    paths_all  = [f"{NPZ_DIR}/{tid}.npz" for tid in tids]
    genres_all = [tracks[tid]['genre_top'] for tid in tids]
    label_map  = {g:i for i,g in enumerate(sorted(set(genres_all)))}
    labels_all = [label_map[g] for g in genres_all]
    
    
    # (4) split 별 경로/레이블 분리
    train_paths  = [p for p,t in zip(paths_all, tids) if tracks[t]['split']=='training']
    train_labels = [labels_all[i] for i,t in enumerate(tids) if tracks[t]['split']=='training']
    val_paths    = [p for p,t in zip(paths_all, tids) if tracks[t]['split']=='validation']
    val_labels   = [labels_all[i] for i,t in enumerate(tids) if tracks[t]['split']=='validation']
    test_paths   = [p for p,t in zip(paths_all, tids) if tracks[t]['split']=='test']
    test_labels  = [labels_all[i] for i,t in enumerate(tids) if tracks[t]['split']=='test']
    
    label_map = { g:i for i,g in enumerate(sorted({v['genre_top'] for v in tracks.values()})) }
    with open('label_map.json', 'w') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    

    # (5) DataLoader
    BS  = 16
    pin = (device.type=='cuda')

    train_loader = DataLoader(LazyHyperImageDataset(train_paths, train_labels),
                              batch_size=BS, shuffle=True,  num_workers=4, pin_memory=pin)

    val_loader   = DataLoader(LazyHyperImageDataset(val_paths,   val_labels),
                              batch_size=BS, shuffle=False, num_workers=4, pin_memory=pin)

    test_loader  = DataLoader(LazyHyperImageDataset(test_paths,  test_labels),
                              batch_size=BS, shuffle=False, num_workers=4, pin_memory=pin)


    # (6) 모델·옵티마이저
    num_classes = len(label_map)
    model        = HyperCNN(num_classes).to(device,non_blocking=True)
    criterion    = nn.CrossEntropyLoss()
    optimizer    = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler    = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3)
    
    # (7) 학습 루프
    scaler = torch.amp.GradScaler()               # 1) AMP 스케일러
    best_f1, early_stop = 0, 0
    history = {}
    NUM_EPOCHS = 30
    
    # outer progress bar 으로 epoch 진행률 보고, ETA 자동 계산
    for epoch in tqdm(range(1, NUM_EPOCHS+1), desc="Epochs", unit="epoch"):
        # --- Train ---
        model.train()
        train_loss = 0.0
        # inner bar 을 batch 단위로
        for x_cpu, y_cpu in tqdm(train_loader,
                                 desc=f"  Train {epoch}/{NUM_EPOCHS}",
                                 leave=False,
                                 unit="batch"):
            # 1) CPU→GPU
            x = x_cpu.to(device, non_blocking=True)
            y = y_cpu.to(device, non_blocking=True)
    
            optimizer.zero_grad()
    
            # 2) Forward + backward with autocast
            with torch.cuda.amp.autocast():            
                out  = model(x)
                loss = criterion(out, y)
    
            scaler.scale(loss).backward()              
            scaler.step(optimizer)
            scaler.update()
    
            train_loss += loss.item() * x.size(0)
    
            # 3) 메모리 해제
            del out, loss, x, y
            torch.cuda.empty_cache()
    
        train_loss /= len(train_loader.dataset)
    
        # --- Validate ---
        model.eval()
        val_preds, val_tgts = [], []
        t1_sum = t3_sum = 0.0
        with torch.no_grad():
            for x_cpu, y_cpu in tqdm(val_loader,
                                     desc=f"  Val   {epoch}/{NUM_EPOCHS}",
                                     leave=False,
                                     unit="batch"):
                x = x_cpu.to(device, non_blocking=True)
                y = y_cpu.to(device, non_blocking=True)
    
                out = model(x)
                t1_sum += topk_accuracy(out, y, 1) * x.size(0)
                t3_sum += topk_accuracy(out, y, 3) * x.size(0)
    
                val_preds.append(out.argmax(1).cpu().numpy())
                val_tgts.append(y.cpu().numpy())
    
                del out, x, y
                torch.cuda.empty_cache()
    
        # metrics
        val_preds = np.concatenate(val_preds)
        val_tgts  = np.concatenate(val_tgts)
        val_top1  = t1_sum / len(val_loader.dataset)
        val_top3  = t3_sum / len(val_loader.dataset)
        val_f1    = f1_score(val_tgts, val_preds, average='macro')
        val_cm    = confusion_matrix(val_tgts, val_preds).tolist()
    
        # scheduler & early stop & save
        scheduler.step(val_f1)
        history[epoch] = {
            'train_loss': train_loss,
            'val_top1':   val_top1,
            'val_top3':   val_top3,
            'val_f1':     val_f1,
            'val_cm':     val_cm,
        }
        
        print(f"Epoch {epoch}/{NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  "
              f"val_top1={val_top1:.3f}  val_top3={val_top3:.3f}  val_f1={val_f1:.3f}")
    
        if val_f1 > best_f1:
            best_f1, early_stop = val_f1, 0
            torch.save(model.state_dict(), 'result/best_hypercnn_2.pth') # result/best_hypercnn.pth 1차 학습
        else:
            early_stop += 1
            if early_stop >= 5:
                print("Early stopping triggered")
                break

    # ----------------------------
    # 8) Test 평가 지표 추가
    # ----------------------------
    model.load_state_dict(torch.load('result/best_hypercnn_2.pth'))
    model.eval()
    test_preds, test_tgts = [], []
    t1_sum, t3_sum = 0, 0

    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            t1_sum += topk_accuracy(out,y,1)*x.size(0)
            t3_sum += topk_accuracy(out,y,3)*x.size(0)
            test_preds.append(out.argmax(1).cpu().numpy())
            test_tgts.append(y.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    test_tgts  = np.concatenate(test_tgts)
    test_top1  = t1_sum / len(test_loader.dataset)
    test_top3  = t3_sum / len(test_loader.dataset)
    test_f1    = f1_score(test_tgts, test_preds, average='macro')
    test_cm    = confusion_matrix(test_tgts, test_preds).tolist()

    # history 및 JSON 저장
    history['test'] = {
        'test_top1': test_top1, 'test_top3': test_top3,
        'test_f1': test_f1, 'test_cm': test_cm
    }
    with open('result/hypercnn_results_2.json', 'w') as f: # result/hypercnn_result.json 1차 학습
        json.dump(history, f, indent=2)

    print(f"\n[Test 결과] Top-1: {test_top1:.4f}, Top-3: {test_top3:.4f}, F1: {test_f1:.4f}")
    

    

if __name__ == "__main__":
    main()
    

    




