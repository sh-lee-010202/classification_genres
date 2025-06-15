# -*- coding: utf-8 -*-
"""
Created on Mon May 26 20:22:22 2025

추론 스크립트: 학습된 5-Fold HyperCNN 모델을 불러와 WAV 파일을 Hyper-image로 변환 후
Top-1 및 Top-3 예측 결과를 파일 단위로 출력합니다.
"""
import os
# ----------------------------
# OpenMP 중복 초기화 오류 우회 설정
# 일부 환경에서 libiomp5md.dll이 중복 로드되어 발생하는 오류를 피하기 위함
# ----------------------------
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import json
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from preprocess.hyperimage_v2 import extract_hyper_image

# ----------------------------
# Dataset 클래스 정의
# ----------------------------
class WavHyperDataset(Dataset):
    """
    WAV 파일 경로 리스트와 레이블을 받아서,
    하이퍼이미지로 변환 후 (1, H, W) 형태의 텐서와 레이블 반환
    """
    def __init__(self, wav_paths, labels, sr=22050, hop_length=512):
        # 전달받은 WAV 경로 및 레이블 저장
        self.wav_paths  = wav_paths
        self.labels     = labels
        self.sr         = sr
        self.hop_length = hop_length

    def __len__(self):
        # 데이터셋 크기: WAV 파일 개수
        return len(self.wav_paths)

    def __getitem__(self, idx):
        # 1) WAV 파일 → Hyper-image 변환
        hyper = extract_hyper_image(
            self.wav_paths[idx],
            sr=self.sr,
            hop_length=self.hop_length
        ).astype(np.float32)
        # 2) NumPy 배열 → PyTorch 텐서로 변환, 채널 차원 추가
        x = torch.from_numpy(hyper).unsqueeze(0)  # (1, H, W)
        # 3) 레이블 텐서 생성
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ----------------------------
# 모델 정의: HyperCNN
# ----------------------------
class HyperCNN(nn.Module):
    """
    하이퍼이미지 분류용 간단한 CNN 모델
    - Conv2d → ReLU → LRN → MaxPool 반복 3회
    - 전역 평균 풀링(AdaptiveAvgPool2d)
    - Flatten → FC(256) → ReLU → Dropout(0.3) → 출력 FC(num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        # 특징 추출부
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.LocalResponseNorm(5), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.LocalResponseNorm(5), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.ReLU(), nn.LocalResponseNorm(5), nn.MaxPool2d(2),
        )
        # 전역 평균 풀링: (batch, 128, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        # 분류기: Flatten 후 FC 레이어
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # → (batch, 128)
            nn.Linear(128, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),   # 클래스 개수에 맞춰 출력
        )

    def forward(self, x):
        # 1) 특징 추출
        x = self.features(x)
        # 2) 전역 평균 풀링
        x = self.global_pool(x)
        # 3) 분류기 통과
        return self.classifier(x)

# ----------------------------
# 앙상블 모델 로드 함수
# ----------------------------
def load_models(paths, device, num_classes):
    """
    학습된 체크포인트(.pth) 리스트를 받아,
    HyperCNN 모델을 로드해 리스트로 반환
    """
    models = []
    for p in paths:
        # 모델 인스턴스 생성 및 device 할당
        m = HyperCNN(num_classes).to(device)
        # state_dict 로드 (weights_only=True로 보안 강화)
        state = torch.load(p, map_location=device, weights_only=True)
        m.load_state_dict(state)
        m.eval()  # 추론 모드로 설정
        models.append(m)
    return models

# ----------------------------
# 추론 및 결과 출력 함수
# ----------------------------
def evaluate_ensemble(models, loader, device, idx2name):
    """
    앙상블된 모델 리스트와 DataLoader를 받아,
    각 WAV 파일에 대해 Top-1 및 Top-3 예측 결과를 출력
    """
    with torch.no_grad():
        for batch in loader:
            # 배치 데이터 분리
            x, y = batch
            x = x.to(device)
            # 모델별 softmax 확률 합산
            prob_sum = None
            for m in models:
                out = m(x)
                p = torch.softmax(out, dim=1)
                prob_sum = p if prob_sum is None else prob_sum + p
            # 평균 확률 계산
            avg_prob = prob_sum / len(models)
            # Top-1, Top-3 인덱스 추출
            top1 = avg_prob.argmax(dim=1).cpu().tolist()
            top3 = avg_prob.topk(3, dim=1)[1].cpu().tolist()
            # 각 샘플별 결과 출력
            for j in range(x.size(0)):
                wav_file   = Path(loader.dataset.wav_paths[j]).name
                true_genre = idx2name[y[j].item()]
                pred1      = idx2name[top1[j]]
                pred3_list = [idx2name[idx] for idx in top3[j]]
                print(f"{wav_file} | True: {true_genre} | Top-1: {pred1} | Top-3: {pred3_list}")

# ----------------------------
# 메인 함수: CSV 로드 → 모델 로드 → 데이터 준비 → 추론 수행
# ----------------------------
if __name__ == "__main__":
    # 1) 장치 설정 (GPU 사용 가능 시 GPU, 아니면 CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2) 라벨 맵 로드 (장르명→정수 매핑)
    with open('label_map_16genres.json', 'r') as f:
        name2idx = json.load(f)
    idx2name   = {v: k for k, v in name2idx.items()}
    num_classes = len(name2idx)

    # 3) 앙상블 모델 체크포인트 경로 지정
    model_paths = [f'./result/best_model_fold{i}_2.pth' for i in range(1,6)]
    models      = load_models(model_paths, device, num_classes)

    # 4) 테스트 메타데이터 CSV 로드
    df = pd.read_csv('./preprocess/track_test.csv')  # 'track_id', 'track_genre_top' 포함
    # track_id를 6자리 문자열로 패딩 후 중간 10개 선택
    padded_ids = df['track_id'].astype(str).apply(lambda x: x.zfill(6)).tolist()
    total = len(padded_ids)
    mid   = total // 2
    start = max(0, mid - 5)
    track_ids = padded_ids[start:start+10]

    # 5) WAV 경로 및 레이블 리스트 생성
    wav_paths = []
    labels    = []
    for tid, genre in zip(track_ids, df['track_genre_top'].tolist()[start:start+10]):
        # 첫 3자리 디렉토리명으로 경로 구성
        path = Path(f"./dataset/fma/data/fma_medium/{tid[:3]}/{tid}.mp3")
        if path.exists():
            wav_paths.append(str(path))
            labels.append(name2idx[genre])
        else:
            print(f"Warning: 파일 없음, 건너뜀 -> {path}")

    # 6) DataLoader 생성 (batch_size=8)
    test_loader = DataLoader(
        WavHyperDataset(wav_paths, labels),
        batch_size=8, shuffle=False, num_workers=0, pin_memory=True
    )

    # 7) 앙상블 추론 및 결과 출력
    evaluate_ensemble(models, test_loader, device, idx2name)



