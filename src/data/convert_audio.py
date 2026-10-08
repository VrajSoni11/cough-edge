"""Resample a dataset to 16 kHz mono WAV.
Usage (project root):  python src\data\convert_audio.py coughvid|coswara|esc50
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import librosa
import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
SR = 16000


def convert(job):
    src, dst = job
    try:
        out = ROOT / dst
        if out.exists():
            return dst, sf.info(out).duration, "skipped"
        y, _ = librosa.load(ROOT / src, sr=SR, mono=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, y, SR, subtype="PCM_16")
        return dst, len(y) / SR, "ok"
    except Exception as e:
        return dst, 0.0, f"error: {e}"


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in {"coughvid", "coswara", "esc50"}:
        sys.exit("Usage: python src\\data\\convert_audio.py coughvid|coswara|esc50")
    m = pd.read_csv(ROOT / f"data/manifests/{name}_manifest.csv")
    if name == "coughvid":
        m = m[m["status"].notna() | (m["n_experts"] > 0)]
    jobs = list(zip(m["raw_path"], m["wav_path"]))
    print(f"{name}: {len(jobs)} files to convert")
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        results = list(tqdm(ex.map(convert, jobs, chunksize=32), total=len(jobs)))
    log = pd.DataFrame(results, columns=["wav_path", "duration_s", "result"])
    log.to_csv(ROOT / f"data/manifests/{name}_convert_log.csv", index=False)
    print(log["result"].value_counts())
    print(log["duration_s"].describe())