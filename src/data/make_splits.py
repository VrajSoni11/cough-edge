"""Create fixed, grouped, stratified splits. Run ONCE.
Run from project root:  python src\data\make_splits.py
"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "data/manifests"
OUT = ROOT / "splits"
OUT.mkdir(exist_ok=True)
SEED = 42

if (OUT / "coughvid_splits.csv").exists() and "--force" not in sys.argv:
    sys.exit("Splits already exist. Re-running would change them; use --force only if you are sure.")

# ---------------- COUGHVID ----------------
cv = pd.read_csv(MAN / "coughvid_manifest.csv")
cv = cv[cv.status.notna() | (cv.n_experts > 0)].copy()
cv["expert_ok"] = (cv.expert_bad_quality == 0) & cv.expert_binary.notna()
cv["self_ok"] = cv.label_self.notna() & cv.is_cough
cv["y_expert"] = cv.expert_binary.where(cv.expert_ok)
cv["y_self"] = cv.label_self.where(cv.self_ok)

n_all = len(cv)
cv = cv[cv.expert_ok | cv.self_ok].copy()
print(f"Labelled recordings: {n_all} | eligible for at least one task: {len(cv)}")


def tag(v):
    return "na" if pd.isna(v) else str(int(v))


cv["strata"] = [f"{tag(a)}_{tag(b)}" for a, b in zip(cv.y_expert, cv.y_self)]


def one_fold(df, n_splits, seed):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, idx = next(sgkf.split(df, df.strata, groups=df.group_proxy))
    return df.index[idx]


test_idx = one_fold(cv, 7, SEED)                         # ~14% test
val_idx = one_fold(cv.drop(index=test_idx), 6, SEED + 1)  # ~14% of total
cv["split"] = "train"
cv.loc[test_idx, "split"] = "test"
cv.loc[val_idx, "split"] = "val"

# leakage check: no proxy group may appear in more than one split
assert (cv.groupby("group_proxy").split.nunique() == 1).all(), "group leakage!"

print("\nRecordings per split:")
print(cv.split.value_counts().reindex(["train", "val", "test"]))
for task, mask, y in [("EXPERT (0=healthy, 1=abnormal)", "expert_ok", "y_expert"),
                      ("SELF (0=healthy, 1=symptomatic)", "self_ok", "y_self")]:
    t = cv[cv[mask]]
    print(f"\n== {task} ==")
    print(pd.crosstab(t.split, t[y]).reindex(["train", "val", "test"]))

cv[["uuid", "split", "expert_ok", "y_expert", "self_ok", "y_self", "wav_path"]].to_csv(
    OUT / "coughvid_splits.csv", index=False)

# ---------------- Coswara (external test) ----------------
cw = pd.read_csv(MAN / "coswara_manifest.csv")
log = pd.read_csv(MAN / "coswara_convert_log.csv")[["wav_path", "duration_s"]]
cw = cw.merge(log, on="wav_path", how="left")
cw = cw[(cw.session_rank == 0) & (cw.duration_s >= 0.5) & cw.label_abnormal.notna()]
subj = cw.drop_duplicates("subject_id")
print(f"\nCoswara external: {len(cw)} files from {cw.subject_id.nunique()} subjects")
print("label_abnormal per subject:\n", subj.label_abnormal.value_counts())
print("label_covid per subject:\n", subj.label_covid.value_counts(dropna=False))
cw[["wav_path", "subject_id", "cough_type", "label_abnormal", "label_covid", "duration_s"]].to_csv(
    OUT / "coswara_external.csv", index=False)

# ---------------- ESC-50 (official folds) ----------------
es = pd.read_csv(MAN / "esc50_manifest.csv")[["filename", "fold", "category", "is_cough", "wav_path"]]
es.to_csv(OUT / "esc50_folds.csv", index=False)
print("\nSaved splits to", OUT)