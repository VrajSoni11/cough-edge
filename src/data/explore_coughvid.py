"""Explore COUGHVID metadata.
Run from project root:  python src\data\explore_coughvid.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/raw/coughvid/public_dataset"
df = pd.read_csv(DATA_DIR / "metadata_compiled.csv")

print("Rows in CSV:", len(df))
missing = [u for u in df.uuid if not (DATA_DIR / f"{u}.wav").exists()]
print("Rows with no matching WAV:", len(missing))

print("\n--- cough_detected ---")
print(df.cough_detected.describe())
for t in (0.5, 0.8, 0.9):
    print(f">= {t}: {(df.cough_detected >= t).sum()}")

print("\n--- SNR ---")
print(df.SNR.describe())

print("\n--- self-reported status ---")
print(df.status.value_counts(dropna=False))

print("\n--- expert annotation ---")
diag_cols = [f"diagnosis_{i}" for i in range(1, 5)]
qual_cols = [f"quality_{i}" for i in range(1, 5)]
n_annot = df[diag_cols].notna().sum(axis=1)
print("Number of expert diagnoses per recording:")
print(n_annot.value_counts().sort_index())
print("\nDiagnosis values (all experts):")
print(pd.concat([df[c] for c in diag_cols]).value_counts())
print("\nQuality values (all experts):")
print(pd.concat([df[c] for c in qual_cols]).value_counts())

print("\n--- demographics ---")
print(df.gender.value_counts(dropna=False))
print(df.age.describe())

print("\n--- status vs cough_detected >= 0.8 ---")
print(pd.crosstab(df.status.fillna("none"), df.cough_detected >= 0.8))

print("\n--- recordings with expert diagnosis AND self-reported status ---")
has_expert = n_annot > 0
print(pd.crosstab(has_expert, df.status.fillna("none")))