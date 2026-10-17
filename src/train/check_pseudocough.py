"""Pseudo-cough check + sensitivity of the main expert task. Grouped CV, train+val only.
Run (main .venv):  python src\train\check_pseudocough.py
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
SEED, N_REPEATS, TAG = 42, 2, "_v2"

idx = pd.read_csv(ROOT / f"data/features/coughvid{TAG}_index.csv")
X_mfcc = np.load(ROOT / f"data/features/coughvid{TAG}_mfccstats.npy")
X_yam = np.hstack([np.load(ROOT / f"data/features/coughvid{TAG}_yamnet_mean.npy"),
                   np.load(ROOT / f"data/features/coughvid{TAG}_yamnet_max.npy")])
man = pd.read_csv(ROOT / "data/manifests/coughvid_manifest.csv")[["uuid", "group_proxy"]]
sevcols = [f"severity_{i}" for i in range(1, 5)]
raw = pd.read_csv(ROOT / "data/raw/coughvid/public_dataset/metadata_compiled.csv")[["uuid"] + sevcols]
df = idx.merge(man, on="uuid", how="left").merge(raw, on="uuid", how="left")
assert len(df) == len(X_mfcc) == len(X_yam) and (df.uuid.values == idx.uuid.values).all()

sev = df[sevcols]
pseudo = (sev == "pseudocough").any(axis=1)
has_sev = sev.notna().any(axis=1)
df["sev_group"] = np.where(pseudo, "pseudocough", np.where(has_sev, "real_cough_rated", "no_severity_info"))

base = (df.expert_ok & df.split.isin(["train", "val"])).values
b = df[base]
print("Severity rating status vs expert label (0 = healthy, 1 = abnormal):")
ct = pd.crosstab(b.sev_group, b.y_expert)
ct["share_abnormal"] = (ct[1.0] / ct.sum(axis=1)).round(3)
print(ct)


def rf():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1,
                                  class_weight="balanced_subsample", random_state=SEED)


def svm():
    return make_pipeline(StandardScaler(), SVC(C=1, class_weight="balanced"))


def cv_auc(make_model, X, y, g):
    out = []
    for r in range(N_REPEATS):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + r)
        for tr, te in sgkf.split(np.zeros(len(y)), y, g):
            m = make_model().fit(X[tr], y[tr])
            s = m.predict_proba(X[te])[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X[te])
            out.append(roc_auc_score(y[te], s))
    return np.array(out)


print("\nMain expert task, with and without pseudocough-flagged clips:")
for label, mask in (("all clips", base), ("excluding pseudocough", base & ~pseudo.values)):
    sel = np.where(mask)[0]
    y = df.loc[sel, "y_expert"].astype(int).values
    g = df.loc[sel, "group_proxy"].values
    print(f"\n  {label}: n={len(sel)}, class counts {np.bincount(y).tolist()}")
    for name, mk, X in (("mfcc RF", rf, X_mfcc), ("yamnet SVM", svm, X_yam), ("yamnet RF", rf, X_yam)):
        a = cv_auc(mk, X[sel], y, g)
        print(f"    {name:11s} AUC {a.mean():.3f} +/- {a.std():.3f}")