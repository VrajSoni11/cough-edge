"""Extract log-Mel and MFCC-statistics features from the 2 s segments.
Run from project root:
    python src\features\extract.py          # v1 segments -> coughvid_*.npy
    python src\features\extract.py _v2      # v2 segments -> coughvid_v2_*.npy
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/features"
FEAT.mkdir(parents=True, exist_ok=True)

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
SR, N_FFT, HOP, N_MELS, FMIN, FMAX, N_MFCC = 16000, 1024, 256, 64, 50, 8000, 40
N_SAMPLES = 32000


def features(seg_path):
    y, sr = sf.read(ROOT / seg_path, dtype="float32")
    assert sr == SR and len(y) == N_SAMPLES, f"unexpected audio: {seg_path}"
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    logmel = librosa.power_to_db(S, ref=1.0, amin=1e-10, top_db=None).astype(np.float32)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=N_MFCC)
    d = librosa.feature.delta(mfcc)
    stats = np.concatenate([mfcc.mean(1), mfcc.std(1), d.mean(1), d.std(1)]).astype(np.float32)
    return logmel, stats


def run(name, index_df):
    paths = index_df["seg_path"].tolist()
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        out = list(tqdm(ex.map(features, paths, chunksize=64), total=len(paths), desc=name + TAG))
    X_mel = np.stack([o[0] for o in out])
    X_stat = np.stack([o[1] for o in out])
    np.save(FEAT / f"{name}{TAG}_logmel.npy", X_mel)
    np.save(FEAT / f"{name}{TAG}_mfccstats.npy", X_stat)
    index_df.to_csv(FEAT / f"{name}{TAG}_index.csv", index=False)   # row i of arrays = row i here
    n_nan = int(np.isnan(X_mel).sum() + np.isnan(X_stat).sum())
    print(f"{name}{TAG}: logmel {X_mel.shape}, mfccstats {X_stat.shape}, NaNs: {n_nan}")


if __name__ == "__main__":
    print("Segment version:", repr(TAG) if TAG else "v1 (no tag)")
    seg = pd.read_csv(ROOT / f"data/manifests/segments{TAG}_index.csv")[["wav_path", "seg_path", "ok"]]

    cv = pd.read_csv(ROOT / "splits/coughvid_splits.csv").merge(seg, on="wav_path", how="left")
    cv = cv[cv["ok"] == True].reset_index(drop=True)
    cw = pd.read_csv(ROOT / "splits/coswara_external.csv").merge(seg, on="wav_path", how="left")
    cw = cw[cw["ok"] == True].reset_index(drop=True)

    print("First segment path:", cv.seg_path.iloc[0])
    print("COUGHVID segments:", len(cv))
    print("Coswara external:", len(cw), "files,", cw.subject_id.nunique(), "subjects")
    print(cw.drop_duplicates("subject_id").label_abnormal.value_counts(), "\n")

    run("coughvid", cv)
    run("coswara", cw)