import pandas as pd
import numpy as np
from pathlib import Path
import random

# npz 파일과 csv 파일을 읽어 딕셔너리로 반환하는 함수 

def dataToDict(csv_path: str, npz_dir: str) -> dict:
    
    #csv_path : FMA metadata CSV 파일 경로 (tracks.csv)
    #npz_dir  : .npz 하이퍼이미지들이 들어있는 폴더 경로
    
    # 1) CSV 읽기
    #    - header=1: 두 번째 줄을 컬럼명으로 사용
    #    - dtype: track_id 를 문자열로 읽기
    df = pd.read_csv(csv_path, dtype={'track_id': str})
    
    # 만약 여전히 'track_id' 컬럼이 없으면 오류 메시지 출력
    if 'track_id' not in df.columns:
        raise KeyError(f"Expected 'track_id' column in {csv_path}, got {df.columns.tolist()}")
    
    base = Path(npz_dir)
    data_dict = {}
    
    for _, row in df.iterrows():
        tid = row['track_id'].zfill(6)          # '134' → '000134'
        npz_file = base / f"{tid}.npz"
        if not npz_file.exists():
            print(f"[Warning] {npz_file.name} not found. skipping.")
            continue
        
        # 하이퍼이미지 로드
        try:
            arr = np.load(npz_file)
            features = arr['hyper'].astype(np.float32)
        except Exception as e:
            print(f"[Error] {tid}: failed to load .npz ({e})")
            continue
        
        # genre_top, set_split 결측 처리
        genre = row.get('track_genre_top', None)
        split = row.get('set_split', None)
        if pd.isna(genre) or pd.isna(split):
            print(f"[Warning] {tid}: missing genre_top or set_split, skipping.")
            continue
        
        data_dict[tid] = {
            'tid':       row['track_id'],       # 원래 숫자형 ID (str)
            'features':  features,              # (820, T) or whatever shape
            'genre_top': genre,                 # e.g. 'rock'
            'split':     split,                 # 'training'/'validation'/'test'
        }
    
    print(f"Loaded {len(data_dict)}/{len(df)} tracks from {csv_path}")
    return data_dict
"""
def dataToDict(csv_path: str, npz_dir: str, max_items: int = None) -> dict:
    df = pd.read_csv(csv_path, dtype={'track_id': str})
    base = Path(npz_dir)

    data_dict = {}
    for i, (_, row) in enumerate(df.iterrows()):
        if max_items and i >= max_items:
            break

        raw_tid    = row['track_id']
        padded_tid = raw_tid.zfill(6)
        npz_file   = base / f"{padded_tid}.npz"
        if not npz_file.is_file():
            continue
        # genre나 split이 없는 경우도 건너뛰기
        if pd.isna(row.get('track_genre_top')) or pd.isna(row.get('set_split')):
            continue

        # 실제 매핑
        with np.load(npz_file) as archive:
            features = archive['hyper'].astype(np.float32)
        data_dict[padded_tid] = {
            'features':  features,
            'genre_top': row['track_genre_top'],
            'split':     row['set_split'],
        }

    print(f"Loaded {len(data_dict)} / {len(df)} tracks (max_items={max_items})")
    return data_dict


if __name__ == "__main__":
    CSV_PATH = './preprocess/track.csv'
    NPZ_DIR  = 'arrays'
    tracks   = dataToDict(CSV_PATH, NPZ_DIR)

    err = 0
    for tid, info in tracks.items():
        if info['features'].shape != (628, 1287):
            print(f"{tid}: {info['features'].shape}")
            err += 1
    print('error count :', err) if err else print('no error')

"""
    # sample_items = random.sample(list(tracks.items()), 5)

    # for tid, info in sample_items:
    #     shape = info['features'].shape
    #     genre = info['genre_top']
    #     split = info['split']
    #     print(tid, shape, genre, split)
