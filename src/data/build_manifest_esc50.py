"""Build the ESC-50 manifest.
Run from project root:  python src\data\build_manifest_esc50.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
m = pd.read_csv(ROOT / "data/raw/esc50/meta/esc50.csv")
m["dataset"] = "esc50"
m["is_cough"] = (m["category"] == "coughing").astype(int)
m["raw_path"] = "data/raw/esc50/audio/" + m["filename"]
m["wav_path"] = "data/wav/esc50/" + m["filename"]
out = m[["filename", "fold", "target", "category", "is_cough", "dataset", "raw_path", "wav_path"]]
out.to_csv(ROOT / "data/manifests/esc50_manifest.csv", index=False)
print(out.is_cough.value_counts())
print(out.fold.value_counts().sort_index())