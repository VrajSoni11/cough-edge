"""Classical baselines + shortcut audit. Grouped CV on train+val only.
Run from project root:  python src\train\train_classical.py
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
SEED = 42
(ROOT / "results/tables").mkdir(parents=True, exist_ok=True)

# ---------- load features + metadata (row i of X == row i of df) ----------
idx = pd.read_csv(ROOT / "data/features/coughvid_index.csv")
X_all = np.load(ROOT / "data/features/coughvid_mfccstats.npy")
man = pd.read_csv(ROOT / "data/manifests/coughvid_manifest.csv")[
    ["uuid", "group_proxy", "SNR", "cough_detected", "age", "gender"]]
seg = pd.read_csv(ROOT / "data/manifests/segments_index.csv")[
    ["wav_path", "orig_dur_s", "trim_dur_s", "peak", "seg_rms_db"]]
df = idx.merge(man, on="uuid", how="left").merge(seg, on="wav_path", how="left")
assert len(df) == len(X_all) and (df.uuid.values == idx.uuid.values).all()
df["gender_code"] = df.gender.map({"male": 0, "female": 1, "other": 2}).fillna(-1)
df["age"] = df.age.fillna(-1)
META = ["orig_dur_s", "trim_dur_s", "peak", "seg_rms_db", "SNR", "cough_detected", "age", "gender_code"]
df["SNR"] = df.SNR.fillna(-99)


def metrics(y, score, pred):
    return {"auc": roc_auc_score(y, score), "ap": average_precision_score(y, score),
            "bal_acc": balanced_accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "sens": recall_score(y, pred, pos_label=1), "spec": recall_score(y, pred, pos_label=0)}


def cv_eval(make_model, X, y, groups):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for tr, te in sgkf.split(X, y, groups):
        m = make_model().fit(X[tr], y[tr])
        score = m.predict_proba(X[te])[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X[te])
        rows.append(metrics(y[te], score, m.predict(X[te])))
    return pd.DataFrame(rows)


def rf():
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                                  class_weight="balanced_subsample", random_state=SEED)


MODELS = {
    "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000)),
    "SVM_C1": lambda: make_pipeline(StandardScaler(), SVC(C=1, class_weight="balanced")),
    "SVM_C10": lambda: make_pipeline(StandardScaler(), SVC(C=10, class_weight="balanced")),
    "RandomForest": rf,
}

results = []
for task, (ok, ycol) in {"expert": ("expert_ok", "y_expert"), "self": ("self_ok", "y_self")}.items():
    sel = np.where(df[ok] & df.split.isin(["train", "val"]))[0]
    y = df.loc[sel, ycol].astype(int).values
    groups = df.loc[sel, "group_proxy"].values
    print(f"\n=== TASK: {task} | n={len(sel)} | positives: {y.mean():.1%} ===")

    s = cv_eval(rf, df.loc[sel, META].values, y, groups)
    print(f"SHORTCUT AUDIT (metadata only, no audio): AUC {s.auc.mean():.3f} +/- {s.auc.std():.3f}")
    results.append({"task": task, "model": "AUDIT_metadata_only", **s.mean().round(4).to_dict()})

    for name, mk in MODELS.items():
        s = cv_eval(mk, X_all[sel], y, groups)
        print(f"{name:13s} AUC {s.auc.mean():.3f}+/-{s.auc.std():.3f} | AP {s.ap.mean():.3f} | "
              f"balAcc {s.bal_acc.mean():.3f} | macroF1 {s.macro_f1.mean():.3f} | "
              f"sens {s.sens.mean():.3f} | spec {s.spec.mean():.3f}")
        results.append({"task": task, "model": name, **s.mean().round(4).to_dict(), "auc_std": round(s.auc.std(), 4)})

pd.DataFrame(results).to_csv(ROOT / "results/tables/baseline_cv.csv", index=False)
print("\nSaved results/tables/baseline_cv.csv")