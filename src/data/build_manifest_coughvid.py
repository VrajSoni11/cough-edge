"""Build the COUGHVID manifest.
Run from project root:  python src\data\build_manifest_coughvid.py
"""
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/coughvid/public_dataset"
OUT = ROOT / "data/manifests"
OUT.mkdir(parents=True, exist_ok=True)

DIAG = [f"diagnosis_{i}" for i in range(1, 5)]
QUAL = [f"quality_{i}" for i in range(1, 5)]
BAD_Q = {"poor", "no_cough"}


def summarize_experts(row):
    diags = [d for d in row[DIAG] if isinstance(d, str)]
    quals = [q for q in row[QUAL] if isinstance(q, str)]
    n = len(diags)
    if n == 0:
        return pd.Series({"n_experts": 0, "expert_diag": None, "expert_binary": np.nan,
                          "expert_agree": np.nan, "expert_bad_quality": np.nan})
    top = Counter(diags).most_common(2)
    diag = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else None   # None = tie
    votes = [0 if d == "healthy_cough" else 1 for d in diags]
    frac = sum(votes) / n
    binary = np.nan if frac == 0.5 else float(frac > 0.5)                  # 1 = abnormal
    bad = float(sum(q in BAD_Q for q in quals) >= len(quals) / 2) if quals else np.nan
    return pd.Series({"n_experts": n, "expert_diag": diag, "expert_binary": binary,
                      "expert_agree": max(frac, 1 - frac), "expert_bad_quality": bad})


df = pd.read_csv(RAW / "metadata_compiled.csv")
df["SNR"] = df["SNR"].replace([np.inf, -np.inf], np.nan)

exp = df.apply(summarize_experts, axis=1)
for c in ["n_experts", "expert_binary", "expert_agree", "expert_bad_quality"]:
    exp[c] = pd.to_numeric(exp[c])

keep = ["uuid", "datetime", "cough_detected", "SNR", "age", "gender",
        "respiratory_condition", "fever_muscle_pain", "status"]   # lat/lon dropped on purpose
df = pd.concat([df[keep], exp], axis=1)

df["label_self"] = df["status"].map({"healthy": 0, "symptomatic": 1})   # COVID-19 / NaN -> NaN
df["is_cough"] = df["cough_detected"] >= 0.8
df["dataset"] = "coughvid"
df["raw_path"] = "data/raw/coughvid/public_dataset/" + df["uuid"] + ".wav"
df["wav_path"] = "data/wav/coughvid/" + df["uuid"] + ".wav"
df.to_csv(OUT / "coughvid_manifest.csv", index=False)

print("Total rows:", len(df))
print("\nSelf-reported binary, cough_detected >= 0.8:")
print(df[df.is_cough].label_self.value_counts())
ex = df[df.n_experts > 0]
print("\nExpert-annotated recordings:", len(ex))
print(ex.expert_diag.value_counts(dropna=False))
clean = ex[(ex.expert_bad_quality == 0) & ex.expert_binary.notna()]
print("\nExpert binary after quality filter (0=healthy, 1=abnormal):", len(clean))
print(clean.expert_binary.value_counts())
print("\nSelf vs expert on the overlap:")
print(pd.crosstab(clean.expert_binary, clean.label_self, dropna=False))