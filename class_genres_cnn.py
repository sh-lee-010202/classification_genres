# -*- coding: utf-8 -*-
"""
Created on Thu May 15 21:57:38 2025

@author: User
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from preprocess import dataToDict
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

# ----------------------------
# 1) 데이터 로드
# ----------------------------
# CSV 파일 경로와 npz 배열 폴더 경로
CSV_PATH = 'tracks.csv'
NPZ_DIR  = 'hyperimage'

# 데이터 딕셔너리 로드
# 반환값: { '000001': {'features': np.ndarray(820,T), 'genre_top': str, 'split': 'training'/'validation'}, ... }
tracks_dict = dataToDict(CSV_PATH, NPZ_DIR)  # :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}

# 학습/검증 리스트 구성
label_map = {g:i for i,g in enumerate(sorted({v['genre_top'] for v in tracks_dict.values()}))}
train_items = [(v['features'], label_map[v['genre_top']]) 
               for v in tracks_dict.values() if v['split']=='training']
val_items   = [(v['features'], label_map[v['genre_top']]) 
               for v in tracks_dict.values() if v['split']=='validation']

# ----------------------------
# 2) Dataset 클래스
# ----------------------------
class NPZDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        hyper, label = self.items[idx]
        # (820, T) → (1,820,T)
        x = torch.from_numpy(hyper).float().unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y

train_ds = NPZDataset(train_items)
val_ds   = NPZDataset(val_items)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

# ----------------------------
# 3) 모델 정의 (일반 CNN 예시)
# ----------------------------
class HyperCNN(nn.Module):
    def __init__(self, num_classes=len(label_map)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.LocalResponseNorm(5),        nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.LocalResponseNorm(5),          nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.LocalResponseNorm(5),           nn.MaxPool2d(2),
        )
        # height: 820→410→205→102, width: T→T/2→T/4→T/8
        example_input = torch.zeros(1,1,820,1287)  # T=1287 고정
        h = self.features(example_input).shape[2]
        w = self.features(example_input).shape[3]
        self.classifier = nn.Sequential(
            nn.Linear(256*h*w, 512),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ----------------------------
# 4) 지표 함수
# ----------------------------
def topk_accuracy(output, target, k=1):
    _, pred = output.topk(k, dim=1)
    return pred.eq(target.view(-1,1)).any(dim=1).float().mean().item()

# ----------------------------
# 5) 학습/평가 루프
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HyperCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

history = {}
best_f1 = 0
early_stop = 0

for epoch in range(1, 31):
    # --- 학습 ---
    model.train()
    total_loss = 0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    train_loss = total_loss / len(train_loader.dataset)

    # --- 평가 ---
    model.eval()
    all_preds, all_tgts = [], []
    top1_sum, top3_sum = 0, 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch} Eval"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            top1_sum += topk_accuracy(out, y, 1) * x.size(0)
            top3_sum += topk_accuracy(out, y, 3) * x.size(0)
            all_preds.append(out.argmax(1).cpu().numpy())
            all_tgts.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    tgts  = np.concatenate(all_tgts)
    top1 = top1_sum / len(val_loader.dataset)
    top3 = top3_sum / len(val_loader.dataset)
    f1   = f1_score(tgts, preds, average='macro')
    cm   = confusion_matrix(tgts, preds).tolist()

    scheduler.step(f1)

    # 모델 저장 / Early Stopping
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), 'best_hypercnn.pth')
        early_stop = 0
    else:
        early_stop += 1
        if early_stop >= 5:
            print("Early stopping.")
            break

    history[epoch] = {
        'train_loss': train_loss,
        'val_top1':   top1,
        'val_top3':   top3,
        'val_f1':     f1,
        'confusion_matrix': cm
    }
    print(f"Epoch {epoch}:", history[epoch])

# 결과 저장
with open('hypercnn_results.json', 'w') as f:
    json.dump(history, f, indent=2)

