"""YAMNet vs MFCC vs metadata, plus SNR-matched control. Grouped CV, train+val only.
Run from project root (main .venv):  python src\train\eval_yamnet.py
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
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
SEED, N_REPEATS, TAG = 42, 2, "_v2"

idx = pd.read_csv(ROOT / f"data/features/coughvid{TAG}_index.csv")
X_mfcc = np.load(ROOT / f"data/features/coughvid{TAG}_mfccstats.npy")
X_yam = np.hstack([np.load(ROOT / f"data/features/coughvid{TAG}_yamnet_mean.npy"),
                   np.load(ROOT / f"data/features/coughvid{TAG}_yamnet_max.npy")])
man = pd.read_csv(ROOT / "data/manifests/coughvid_manifest.csv")[
    ["uuid", "group_proxy", "SNR", "age", "gender", "datetime"]]
seg = pd.read_csv(ROOT / f"data/manifests/segments{TAG}_index.csv")[
    ["wav_path", "orig_dur_s", "peak", "zero_frac", "seg_rms_db"]]
df = idx.merge(man, on="uuid", how="left").merge(seg, on="wav_path", how="left")
assert len(df) == len(X_mfcc) == len(X_yam) and (df.uuid.values == idx.uuid.values).all()

dt = pd.to_datetime(df.datetime, utc=True, errors="coerce")
df["days"] = ((dt - dt.min()).dt.total_seconds() / 86400).fillna(-1)
df["gender_code"] = df.gender.map({"male": 0, "female": 1, "other": 2}).fillna(-1)
df["age"] = df.age.fillna(-1)
df["SNR_raw"] = df.SNR.replace([np.inf, -np.inf], np.nan)
df["SNR"] = df.SNR_raw.fillna(-99)
META = ["orig_dur_s", "peak", "seg_rms_db", "zero_frac", "SNR", "age", "gender_code", "days"]


def rf():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1,
                                  class_weight="balanced_subsample", random_state=SEED)


def lr(C):
    return lambda: make_pipeline(StandardScaler(),
                                 LogisticRegression(C=C, class_weight="balanced", max_iter=3000))


def svm():
    return make_pipeline(StandardScaler(), SVC(C=1, class_weight="balanced", probability=False))


def cv_auc(make_model, X, y, g, repeats=N_REPEATS, seed=SEED):
    out = []
    for r in range(repeats):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + r)
        for tr, te in sgkf.split(np.zeros(len(y)), y, g):
            m = make_model().fit(X[tr], y[tr])
            s = m.predict_proba(X[te])[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X[te])
            out.append(roc_auc_score(y[te], s))
    return np.array(out)


rows = []
for task, (ok, ycol) in {"expert": ("expert_ok", "y_expert"), "self": ("self_ok", "y_self")}.items():
    sel = np.where(df[ok] & df.split.isin(["train", "val"]))[0]
    y = df.loc[sel, ycol].astype(int).values
    g = df.loc[sel, "group_proxy"].values
    Xm = df.loc[sel, META].values
    print(f"\n{'=' * 8} TASK: {task} | n={len(sel)} | class counts {np.bincount(y).tolist()} {'=' * 8}")

    sets = {"mfcc": X_mfcc[sel], "yamnet": X_yam[sel], "meta": Xm, "yamnet+meta": np.hstack([X_yam[sel], Xm])}
    models = {"LogReg": lr(0.01), "RF": rf}
    if task == "expert":
        models["SVM"] = svm                       # SVM on the large self task is too slow
    res = {}
    print("\nA) Same folds for every row:")
    for sname, X in sets.items():
        for mname, mk in models.items():
            if sname == "meta" and mname == "LogReg":
                mk = lr(0.1)
            a = cv_auc(mk, X, y, g)
            res[(sname, mname)] = a
            print(f"   {sname:12s} {mname:7s} {a.mean():.3f} +/- {a.std():.3f}")
            rows.append({"task": task, "analysis": "sets", "set": sname, "model": mname,
                         "auc": a.mean(), "std": a.std()})
    print("\n   Paired AUC gain (positive = first is better):")
    for mname in models:
        for a_, b_ in (("yamnet", "mfcc"), ("yamnet+meta", "meta"), ("yamnet", "meta")):
            if (a_, mname) in res and (b_, mname) in res:
                d = res[(a_, mname)] - res[(b_, mname)]
                print(f"   {mname:7s} {a_:12s} - {b_:5s}: {d.mean():+.3f} (better in {(d > 0).sum()}/{len(d)} folds)")

    if task == "expert":
        print("\nB) SNR-matched control (class ratio equalised inside each SNR bin):")
        snr = df.loc[sel, "SNR_raw"].values
        bins = pd.qcut(pd.Series(snr), 5, labels=False, duplicates="drop").values
        rec = {k: [] for k in ("snr_only", "mfcc", "yamnet", "meta")}
        sizes = []
        for draw in range(5):
            rng = np.random.default_rng(SEED + draw)
            keep = []
            for b in pd.Series(bins).dropna().unique():
                i0 = np.where((bins == b) & (y == 0))[0]
                i1 = np.where((bins == b) & (y == 1))[0]
                n = min(len(i0), len(i1))
                keep += list(rng.choice(i0, n, replace=False)) + list(rng.choice(i1, n, replace=False))
            keep = np.array(keep)
            sizes.append(len(keep))
            yk, gk = y[keep], g[keep]
            rec["snr_only"] += list(cv_auc(rf, df.loc[sel, ["SNR"]].values[keep], yk, gk, 1, SEED + draw))
            rec["mfcc"] += list(cv_auc(lr(0.01), X_mfcc[sel][keep], yk, gk, 1, SEED + draw))
            rec["yamnet"] += list(cv_auc(lr(0.01), X_yam[sel][keep], yk, gk, 1, SEED + draw))
            rec["meta"] += list(cv_auc(rf, Xm[keep], yk, gk, 1, SEED + draw))
        print(f"   matched subset size per draw: {sizes} (balanced 50/50)")
        for k, v in rec.items():
            v = np.array(v)
            print(f"   {k:9s} AUC {v.mean():.3f} +/- {v.std():.3f}")
            rows.append({"task": task, "analysis": "snr_matched", "set": k, "model": "RF/LogReg",
                         "auc": v.mean(), "std": v.std()})

pd.DataFrame(rows).to_csv(ROOT / "results/tables/eval_yamnet.csv", index=False)
print("\nSaved results/tables/eval_yamnet.csv")