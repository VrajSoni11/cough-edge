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

rows = []
for date_dir in sorted(p for p in EXTRACTED.iterdir() if p.is_dir()):
    for subj in date_dir.iterdir():
        if not subj.is_dir():
            continue
        for kind in ("heavy", "shallow"):
            f = subj / f"cough-{kind}.wav"
            if f.exists():
                rows.append({"subject_id": subj.name, "rec_date": date_dir.name,
                             "cough_type": kind,
                             "raw_path": f.relative_to(ROOT).as_posix(),
                             "wav_path": f"data/wav/coswara/{subj.name}_{kind}.wav"})
files = pd.DataFrame(rows)
df = files.merge(meta, on="subject_id", how="left")
df["dataset"] = "coswara"


def abnormal(s):      # provisional mapping, check the printed values first
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

subj = df.drop_duplicates("subject_id")
print("Cough files:", len(df), "| subjects with audio:", df.subject_id.nunique())
print("Subjects in combined_data.csv:", meta.subject_id.nunique())
print("Audio subjects missing from metadata:", subj.covid_status.isna().sum())
print("Duplicate output paths:", df.wav_path.duplicated().sum())
print("\ncovid_status (per subject):")
print(subj.covid_status.value_counts(dropna=False))
print("\nlabel_abnormal (per subject):")
print(subj.label_abnormal.value_counts(dropna=False))
print("\nlabel_covid (per subject):")
print(subj.label_covid.value_counts(dropna=False))