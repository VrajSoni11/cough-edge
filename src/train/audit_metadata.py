"""What drives the metadata shortcut, and does audio add value on top of it?
Run from project root:  python src\train\audit_metadata.py
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
SEED, N_REPEATS = 42, 2

# ---------- load (row i of X_audio == row i of df) ----------
idx = pd.read_csv(ROOT / "data/features/coughvid_index.csv")
X_audio = np.load(ROOT / "data/features/coughvid_mfccstats.npy")
man = pd.read_csv(ROOT / "data/manifests/coughvid_manifest.csv")[
    ["uuid", "group_proxy", "SNR", "cough_detected", "age", "gender", "datetime"]]
seg = pd.read_csv(ROOT / "data/manifests/segments_index.csv")[
    ["wav_path", "orig_dur_s", "trim_dur_s", "peak", "seg_rms_db"]]
df = idx.merge(man, on="uuid", how="left").merge(seg, on="wav_path", how="left")
assert len(df) == len(X_audio) and (df.uuid.values == idx.uuid.values).all()

dt = pd.to_datetime(df.datetime, utc=True, errors="coerce")
df["days"] = ((dt - dt.min()).dt.total_seconds() / 86400).fillna(-1)
df["gender_code"] = df.gender.map({"male": 0, "female": 1, "other": 2}).fillna(-1)
df["age"] = df.age.fillna(-1)
df["SNR"] = df.SNR.fillna(-99)

GROUPS = {"length": ["orig_dur_s", "trim_dur_s"], "loudness": ["peak", "seg_rms_db"],
          "snr": ["SNR"], "cough_score": ["cough_detected"],
          "demographics": ["age", "gender_code"], "time(date)": ["days"]}
META = sum(GROUPS.values(), [])
DEMO = GROUPS["demographics"]


def rf():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1,
                                  class_weight="balanced_subsample", random_state=SEED)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))


def cv_auc(make_model, X, y, g):
    out = []
    for r in range(N_REPEATS):           # same seeds -> identical folds across feature sets (paired)
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + r)
        for tr, te in sgkf.split(np.zeros(len(y)), y, g):
            m = make_model().fit(X[tr], y[tr])
            out.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.array(out)


rows = []
for task, (ok, ycol) in {"expert": ("expert_ok", "y_expert"), "self": ("self_ok", "y_self")}.items():
    sel = np.where(df[ok] & df.split.isin(["train", "val"]))[0]
    y = df.loc[sel, ycol].astype(int).values
    g = df.loc[sel, "group_proxy"].values
    print(f"\n{'=' * 8} TASK: {task} | n={len(sel)} | positives {y.mean():.1%} {'=' * 8}")

    print("\nA) Each metadata group ALONE (RandomForest, grouped CV AUC):")
    for name, cols in GROUPS.items():
        a = cv_auc(rf, df.loc[sel, cols].values, y, g)
        print(f"   {name:13s} {a.mean():.3f} +/- {a.std():.3f}")
        rows.append({"task": task, "analysis": "group_alone", "set": name, "model": "RF",
                     "auc": a.mean(), "std": a.std()})

    Xa, Xm, Xd = X_audio[sel], df.loc[sel, META].values, df.loc[sel, DEMO].values
    sets = {"audio": Xa, "meta": Xm, "demo": Xd,
            "audio+meta": np.hstack([Xa, Xm]), "audio+demo": np.hstack([Xa, Xd])}
    res = {}
    print("\nB) Incremental value (same folds for every row):")
    for sname, X in sets.items():
        for mname, mk in (("RF", rf), ("LogReg", lr)):
            a = cv_auc(mk, X, y, g)
            res[(sname, mname)] = a
            print(f"   {sname:11s} {mname:6s} {a.mean():.3f} +/- {a.std():.3f}")
            rows.append({"task": task, "analysis": "incremental", "set": sname, "model": mname,
                         "auc": a.mean(), "std": a.std()})
    print("\n   Paired AUC gain from adding audio (positive = audio helps):")
    for mname in ("RF", "LogReg"):
        for a_, b_ in (("audio+meta", "meta"), ("audio+demo", "demo")):
            d = res[(a_, mname)] - res[(b_, mname)]
            print(f"   {mname:6s} {a_:10s} - {b_:5s}: {d.mean():+.3f}  (better in {(d > 0).sum()}/{len(d)} folds)")

pd.DataFrame(rows).to_csv(ROOT / "results/tables/audit_metadata.csv", index=False)
print("\nSaved results/tables/audit_metadata.csv")