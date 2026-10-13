"""YAMNet embeddings for the 2 s segments. Run in the TF venv (.venv-tf):
    python src\features\extract_yamnet.py _v2
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/features"
TAG = sys.argv[1] if len(sys.argv) > 1 else "_v2"
HANDLE = "https://tfhub.dev/google/yamnet/1"

model = hub.load(HANDLE)


def embed(seg_path):
    y, sr = sf.read(ROOT / seg_path, dtype="float32")
    assert sr == 16000 and y.ndim == 1, f"unexpected audio: {seg_path}"
    _, emb, _ = model(tf.constant(y))          # (n_frames, 1024)
    emb = emb.numpy()
    return emb.mean(0), emb.max(0), emb.shape[0]


for name in ("coughvid", "coswara"):
    index = pd.read_csv(FEAT / f"{name}{TAG}_index.csv")
    out = [embed(p) for p in tqdm(index.seg_path, desc=name)]
    mean = np.stack([o[0] for o in out]).astype(np.float32)
    mx = np.stack([o[1] for o in out]).astype(np.float32)
    frames = np.array([o[2] for o in out])
    np.save(FEAT / f"{name}{TAG}_yamnet_mean.npy", mean)
    np.save(FEAT / f"{name}{TAG}_yamnet_max.npy", mx)
    print(f"{name}: mean {mean.shape}, max {mx.shape}, NaNs {int(np.isnan(mean).sum())}, "
          f"frames per clip: {np.unique(frames, return_counts=True)}")