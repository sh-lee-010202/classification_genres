# -*- coding: utf-8 -*-
"""
Created on Wed May 21 15:28:03 2025

@author: User
"""


import warnings
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt

# ————————————— 경고 필터링 —————————————
warnings.filterwarnings("ignore",
                        message="Cannot read next header, a one-frame stream",
                        module="mpg123")
warnings.filterwarnings("ignore",
                        category=UserWarning,
                        module="librosa")

def safe_load(audio_path: str, sr: int, duration: float = None):

    #soundfile 먼저 시도, 실패하면 librosa.load fallback.
    try:
        y, orig_sr = sf.read(audio_path, dtype='float32')
        if y.size == 0:
            raise RuntimeError("Empty audio")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr, sr)
        return y, sr
    except Exception as e1:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return librosa.load(audio_path, sr=sr, duration=duration)
        except Exception as e2:
            raise RuntimeError(f"soundfile_error={e1!r}, librosa_error={e2!r}")

def extract_hyper_image(
    audio_path: str,
    sr: int = 22050,
    hop_length: int = 512,
):
    # 0. 오디오 로드 & STFT
    y, sr = safe_load(audio_path, sr=sr, duration=30.0)
    D = librosa.stft(y, hop_length=hop_length, center=False)

    features = {}

    # 1. Chromagram (12차원)
    C = np.abs(D)**2
    features['Chromagram'] = librosa.feature.chroma_stft(
        S=C, sr=sr, hop_length=hop_length, n_chroma=12
    )

    # 2. MFCC (40차원)
    features['MFCC'] = librosa.feature.mfcc(
        S=librosa.power_to_db(C), n_mfcc=40
    )

    # 3. Mel-Spectrogram (128차원)
    melspec = librosa.feature.melspectrogram(
        S=C, sr=sr, n_mels=128
    )
    features['Mel-Spectrogram'] = librosa.power_to_db(melspec, ref=np.max)

    # 4. Tonnetz (6차원)
    harmonic, _ = librosa.decompose.hpss(D, margin=(1.2,1.0)) # default = margin(1.0,1.0) / v2: margin=(1.2,1.0)
    y_harm = librosa.istft(harmonic, hop_length=hop_length, center=False)
    features['Tonnetz'] = librosa.feature.tonnetz(y=y_harm, sr=sr)

    # 5. Spectral Contrast (6차원)
    spec_contrast = librosa.feature.spectral_contrast(
        S=np.abs(D),
        sr=sr,
        n_bands=6,
        quantile=0.05 # default=0.02 / v2: quantile=0.05
    )
    features['Spectral Contrast'] = spec_contrast[:-1, :]  # shape (8, T)

    # 6/7. HPSS → Harmonic & Percussive Mel-Spec (각 64차원)
    D_harm, D_perc = librosa.decompose.hpss(D)
    harm_mel = librosa.feature.melspectrogram(
        S=np.abs(D_harm)**2, sr=sr, n_mels=64
    )
    perc_mel = librosa.feature.melspectrogram(
        S=np.abs(D_perc)**2, sr=sr, n_mels=64
    )
    features['HPSS-Harmonic Mel-Spec']  = librosa.power_to_db(harm_mel, ref=np.max)
    features['HPSS-Percussive Mel-Spec'] = librosa.power_to_db(perc_mel, ref=np.max)

    # 8. Fourier Tempogram (192차원)
    temp = librosa.feature.fourier_tempogram(
        y=y, sr=sr, hop_length=hop_length, win_length=384
    )
    temp_mag = np.abs(temp)[:192, :]
    features['Tempogram'] = temp_mag

    # ————————————————
    # 공통 시간축 길이 자르기
    min_frames = 1287
    for name in features:
        features[name] = features[name][:, :min_frames]

    # 기대 높이 매핑 (요청하신 순서)
    expected = {
        'Chromagram':                12,  # 화성 정보 (조성/코드)
        'Tonnetz':                    6,  # 화음 간 관계
        'MFCC':                      40,  # 저차원 스펙트럼 요약
        'Mel-Spectrogram':          128,  # 전체 멜 스펙트럼
        'HPSS-Harmonic Mel-Spec':    64,  # 하모닉 멜 대역
        'HPSS-Percussive Mel-Spec':  64,  # 퍼커시브 멜 대역
        'Spectral Contrast':          6,  # 대역별 대비
        'Tempogram':                192,  # 리듬/템포 패턴
    }

    # 차원 검증
    for name, mat in features.items():
        h, t = mat.shape
        e = expected[name]
        assert h == e, f"{name} height mismatch: got {h}, expected {e}"
        assert t == min_frames, f"{name} time mismatch: got {t}, expected {min_frames}"

    # 최종 수직 결합 (요청하신 순서)
    hyper_image = np.vstack([
        features['Chromagram'],
        features['Tonnetz'],
        features['MFCC'],
        features['Mel-Spectrogram'],
        features['HPSS-Harmonic Mel-Spec'],
        features['HPSS-Percussive Mel-Spec'],
        features['Spectral Contrast'],
        features['Tempogram'],
    ])

    return hyper_image

if __name__ == "__main__":
    audio_path = "./fma_small/000/000010.mp3" # input audio
    hyper = extract_hyper_image(audio_path)
    print("Hyper-image shape:", hyper.shape)  

    # 시각화
    plt.figure(figsize=(12, 6))
    plt.imshow(hyper, 
               aspect='auto', 
               #origin='lower', 
               cmap='magma',      # 
               vmin=-100, vmax=100  # dB 단위 범위(논문 Fig.3 참조) :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
    )
    plt.title("Audio Hyper-Image")
    plt.xlabel("Time frames")
    plt.ylabel("Feature bins")
    plt.colorbar(label="Amplitude (dB)")
    plt.tight_layout()
    plt.show()
    # 수정 사항
    # Tempogram 193 -> 384: fourier_tempogram에 맞게 bin 높이를 193으로 줄이고 1을 잘라 192로 만듦
    # fourier_tempogram의 값이 복소수로 출력되므로 np.abs로 절댓값으로 처리함 
    # T(시간 가로축)이 각 특징마다 같은 값을 갖도록: 모든 특징의 프레임을 1287이 되도록 자름
    # 최종 출력 이미지 높이가 628
