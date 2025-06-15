# 데이터셋 내의 모든 음원 파일에 대한 하이퍼이미지를 생성하고,
# 프로젝트 폴더에 arrays 디렉토리 생성 후 해당 디렉토리 내에 각 음원별로 npz 파일로 저장

import os
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt

from preprocess.hyperimage_v2 import extract_hyper_image

# 1) 000 ~ 155 폴더가 들어있는 fma_medium 폴더 지정
inputdir = "./dataset/fma/data/fma_medium"
#          ├── 000
#          ├── 001
#          ├── …
#          └── 155

# 2) 하이퍼이미지 출력 디렉토리
outputdir = "hyperimage_16genres"

#——————————————

def process_one(audio_path: Path, base_out: Path):
    
    track_id = audio_path.stem
    try:
        hyper = extract_hyper_image(str(audio_path))

        # 배열 저장
        arr_dir = base_out 
        arr_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arr_dir / f"{track_id}.npz", hyper=hyper.astype(np.float32))

        return True, track_id

    except Exception as e:
        return False, f"{track_id}: {repr(e)}"

def main():
    DATASET_DIR = Path(inputdir)
    BASE_OUT_DIR = Path(outputdir) 

    # mp3 파일 리스트
    all_files = list(DATASET_DIR.rglob("*.mp3"))
    print(f"총 {len(all_files)}개 파일 처리 시작")

    n_workers = max(1, os.cpu_count() - 1)
    print(f"n_workers: {n_workers}")
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        futures = {
            exe.submit(process_one, p, BASE_OUT_DIR): p
            for p in all_files
        }
        for future in tqdm(as_completed(futures),
                           total=len(futures), desc="전체 처리"):
            ok, info = future.result()
            if not ok:
                print("[Error]", info)

if __name__ == "__main__":
    main()


# 전체 파일 25000 개 중 21개 파일이 깨져있음