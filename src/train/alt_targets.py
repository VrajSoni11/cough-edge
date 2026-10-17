"""Alternative expert targets: cough type, symptoms, diagnosis sub-types.
Grouped CV on train+val only.   Run (main .venv):  python src\train\alt_targets.py
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
SEED, N_REPEATS, TAG, MIN_PER_CLASS = 42, 2, "_v2", 80

raw = pd.read_csv(ROOT / "data/raw/coughvid/public_dataset/metadata_compiled.csv")
idx = pd.read_csv(ROOT / f"data/features/coughvid{TAG}_index.csv")
X_mfcc = np.load(ROOT / f"data/features/coughvid{TAG}_mfccstats.npy")
X_yam = np.hstack([np.load(ROOT / f"data/features/coughvid{TAG}_yamnet_mean.npy"),
                   np.load(ROOT / f"data/features/coughvid{TAG}_yamnet_max.npy")])
man = pd.read_csv(ROOT / "data/manifests/coughvid_manifest.csv")[
    ["uuid", "group_proxy", "SNR", "age", "gender", "datetime", "n_experts", "expert_bad_quality"]]
seg = pd.read_csv(ROOT / f"data/manifests/segments{TAG}_index.csv")[
    ["wav_path", "orig_dur_s", "peak", "zero_frac", "seg_rms_db"]]
fields = [f"{p}_{i}" for p in ("cough_type", "severity", "dyspnea", "wheezing", "stridor",
                               "choking", "congestion", "diagnosis") for i in range(1, 5)]
df = (idx.merge(man, on="uuid", how="left").merge(seg, on="wav_path", how="left")
         .merge(raw[["uuid"] + fields], on="uuid", how="left"))
assert len(df) == len(X_mfcc) == len(X_yam) and (df.uuid.values == idx.uuid.values).all()

dt = pd.to_datetime(df.datetime, utc=True, errors="coerce")
df["days"] = ((dt - dt.min()).dt.total_seconds() / 86400).fillna(-1)
df["gender_code"] = df.gender.map({"male": 0, "female": 1, "other": 2}).fillna(-1)
df["age"] = df.age.fillna(-1)
df["SNR"] = df.SNR.replace([np.inf, -np.inf], np.nan).fillna(-99)
META = ["orig_dur_s", "peak", "seg_rms_db", "zero_frac", "SNR", "age", "gender_code", "days"]

base = (df.split.isin(["train", "val"]) & (df.n_experts > 0) & (df.expert_bad_quality == 0)).values
print("Base set (train+val, expert-annotated, quality ok):", base.sum())

print("\n--- raw values of expert fields (annotator 1), base set ---")
for c in ("cough_type_1", "severity_1", "dyspnea_1", "wheezing_1", "stridor_1", "choking_1", "congestion_1"):
    print(f"\n{c}:\n", df.loc[base, c].value_counts(dropna=False).to_string())


def p_bool(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    return 1.0 if s in ("true", "1", "1.0", "yes") else 0.0 if s in ("false", "0", "0.0", "no") else np.nan


def p_map(pos, neg):
    return lambda v: np.nan if pd.isna(v) else (
        1.0 if str(v).strip().lower() == pos else 0.0 if str(v).strip().lower() == neg else np.nan)


def p_diag(pos):
    return lambda v: np.nan if pd.isna(v) else (
        1.0 if v == pos else 0.0 if v == "healthy_cough" else np.nan)


TARGETS = {
    "wet_vs_dry": ("cough_type", p_map("wet", "dry")),
    "severe_vs_mild": ("severity", p_map("severe", "mild")),
    "dyspnea": ("dyspnea", p_bool), "wheezing": ("wheezing", p_bool), "stridor": ("stridor", p_bool),
    "choking": ("choking", p_bool), "congestion": ("congestion", p_bool),
    "covid_vs_healthy": ("diagnosis", p_diag("COVID-19")),
    "upper_vs_healthy": ("diagnosis", p_diag("upper_infection")),
    "lower_vs_healthy": ("diagnosis", p_diag("lower_infection")),
    "obstructive_vs_healthy": ("diagnosis", p_diag("obstructive_disease")),
}


def majority(prefix, parser):
    v = np.column_stack([df[f"{prefix}_{i}"].map(parser).values for i in range(1, 5)]).astype(float)
    m = np.nanmean(v, axis=1)
    lab = np.full(len(m), np.nan)
    lab[m > 0.5] = 1.0
    lab[m < 0.5] = 0.0
    return lab


def rf():
    return RandomForestClassifier(n_estimators=150, min_samples_leaf=3, n_jobs=-1,
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


rows = []
for name, (prefix, parser) in TARGETS.items():
    lab = majority(prefix, parser)
    sel = np.where(base & ~np.isnan(lab))[0]
    y = lab[sel].astype(int)
    cnt = np.bincount(y, minlength=2)
    print(f"\n=== {name}: n={len(sel)}, class counts [0, 1] = {cnt.tolist()} ===")
    if len(sel) == 0 or cnt.min() < MIN_PER_CLASS:
        print("   skipped: fewer than", MIN_PER_CLASS, "in the smaller class")
        continue
    g = df.loc[sel, "group_proxy"].values
    res = {"mfcc_RF": cv_auc(rf, X_mfcc[sel], y, g), "yamnet_SVM": cv_auc(svm, X_yam[sel], y, g),
           "yamnet_RF": cv_auc(rf, X_yam[sel], y, g), "meta_RF": cv_auc(rf, df.loc[sel, META].values, y, g)}
    for k, v in res.items():
        print(f"   {k:11s} {v.mean():.3f} +/- {v.std():.3f}")
    best_yam = max(res["yamnet_SVM"].mean(), res["yamnet_RF"].mean())
    rows.append({"target": name, "n": len(sel), "n_pos": int(cnt[1]), "n_neg": int(cnt[0]),
                 **{k: round(v.mean(), 3) for k, v in res.items()}, "best_yamnet": round(best_yam, 3),
                 "yamnet_minus_meta": round(best_yam - res["meta_RF"].mean(), 3)})

summary = pd.DataFrame(rows).sort_values("best_yamnet", ascending=False)
print("\n=== SUMMARY (decision rule: best_yamnet >= 0.70, yamnet_minus_meta >= 0.05, both classes >= 100) ===")
print(summary.to_string(index=False))
summary.to_csv(ROOT / "results/tables/alt_targets.csv", index=False)
print("\nSaved results/tables/alt_targets.csv")