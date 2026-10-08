"""Build the Coswara cough manifest (one row per cough recording).
Run from project root:  python src\data\build_manifest_coswara.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXTRACTED = ROOT / "data/raw/coswara_extracted"
META = ROOT / "data/raw/coswara/combined_data.csv"
OUT = ROOT / "data/manifests"
OUT.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(META).drop(columns=["l_l"])          # drop locality (privacy)
meta = meta.rename(columns={"id": "subject_id", "a": "age", "g": "gender",
                            "l_c": "country", "l_s": "state"})
n_before = len(meta)
meta = meta.drop_duplicates("subject_id")
print("Duplicate metadata rows dropped:", n_before - len(meta))

rows = []
for date_dir in sorted(p for p in EXTRACTED.iterdir() if p.is_dir()):
    for subj in date_dir.iterdir():
        if not subj.is_dir():
            continue
        for kind in ("heavy", "shallow"):
            f = subj / f"cough-{kind}.wav"
            if f.exists():
                rows.append({
                    "subject_id": subj.name, "rec_date": date_dir.name, "cough_type": kind,
                    "raw_path": f.relative_to(ROOT).as_posix(),
                    "wav_path": f"data/wav/coswara/{date_dir.name}_{subj.name}_{kind}.wav"})

files = pd.DataFrame(rows).sort_values(["subject_id", "cough_type", "rec_date"])
files["session_rank"] = files.groupby(["subject_id", "cough_type"]).cumcount()   # 0 = earliest
df = files.merge(meta, on="subject_id", how="left")
df["dataset"] = "coswara"


def abnormal(s):     # provisional mapping, documented in the paper
    if s == "healthy":
        return 0.0
    if isinstance(s, str) and (s.startswith("positive") or s == "resp_illness_not_identified"):
        return 1.0
    return float("nan")


def covid(s):
    if s == "healthy":
        return 0.0
    if isinstance(s, str) and s.startswith("positive"):
        return 1.0
    return float("nan")


df["label_abnormal"] = df["covid_status"].map(abnormal)
df["label_covid"] = df["covid_status"].map(covid)
df.to_csv(OUT / "coswara_manifest.csv", index=False)

multi = df[df.session_rank > 0]
print("\nCough files:", len(df), "| subjects:", df.subject_id.nunique())
print("Duplicate wav paths (must be 0):", df.wav_path.duplicated().sum())
print("Extra sessions (session_rank > 0):", len(multi),
      "| subjects affected:", multi.subject_id.nunique())
first = df[df.session_rank == 0]
print("Files with session_rank == 0:", len(first))
subj = df.drop_duplicates("subject_id")
print("\nlabel_abnormal per subject:\n", subj.label_abnormal.value_counts(dropna=False))
print("\nlabel_covid per subject:\n", subj.label_covid.value_counts(dropna=False))