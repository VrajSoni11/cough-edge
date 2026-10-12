"""Padding-free segmentation: 2 s window of REAL audio centred on the loudest point.
Run from project root:  python src\data\segment_v2.py
"""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SR, WIN, HOP = 16000, 32000, 256
MIN_PEAK = 1e-3


def segment_one(job):
    wav_path, dataset = job
    seg_path = wav_path.replace("data/wav/", "data/segments_v2/")
    row = {"wav_path": wav_path, "seg_path": seg_path, "dataset": dataset, "ok": False, "reason": ""}
    try:
        y, _ = sf.read(ROOT / wav_path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        row["orig_dur_s"] = len(y) / SR
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        row["peak"] = peak
        if peak < MIN_PEAK:
            row["reason"] = "silent"
            return row
        if len(y) < WIN:
            row["reason"] = "shorter_than_window"
            return row
        rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
        center = int(np.argmax(rms)) * HOP
        start = max(0, min(center - WIN // 2, len(y) - WIN))
        seg = y[start:start + WIN]
        row["zero_frac"] = float(np.mean(np.abs(seg) < 1e-4))      # exact digital silence in window
        seg = seg / (np.max(np.abs(seg)) + 1e-9)
        out = ROOT / seg_path
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, seg, SR, subtype="PCM_16")
        row["seg_rms_db"] = float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9))
        row["win_start_s"] = start / SR
        row["ok"] = True
    except Exception as e:
        row["reason"] = f"error: {e}"
    return row


if __name__ == "__main__":
    cv = pd.read_csv(ROOT / "splits/coughvid_splits.csv")
    cw = pd.read_csv(ROOT / "splits/coswara_external.csv")
    jobs = [(p, "coughvid") for p in cv.wav_path] + [(p, "coswara") for p in cw.wav_path]
    print("Recordings to segment:", len(jobs))
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        rows = list(tqdm(ex.map(segment_one, jobs, chunksize=32), total=len(jobs)))
    idx = pd.DataFrame(rows)
    idx.to_csv(ROOT / "data/manifests/segments_v2_index.csv", index=False)
    print("\nOK vs failed per dataset:")
    print(idx.groupby("dataset").ok.value_counts())
    print("\nFailure reasons:\n", idx[~idx.ok].reason.value_counts())
    print("\nzero_frac (digital silence inside window):")
    print(idx[idx.ok].groupby("dataset").zero_frac.describe().round(3))