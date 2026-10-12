"""Padding control: is the audio signal just clip length / padding in disguise?
Run from project root:  python src\train\audit_padding.py
"""
import warnings
from pathlib import Path
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
SEED, N_REPEATS, SR, HOP, WIN = 42, 2, 16000, 256, 32000

idx = pd.read_csv(ROOT / "data/features/coughvid_index.csv")
X_full = np.load(ROOT / "data/features/coughvid_mfccstats.npy")
logmel = np.load(ROOT / "data/features/coughvid_logmel.npy", mmap_mode="r")
man = pd.read_csv(ROOT / "data/manifests/coughvid_manifest.csv")[
    ["uuid", "group_proxy", "SNR", "cough_detected", "age", "gender", "datetime"]]
seg = pd.read_csv(ROOT / "data/manifests/segments_index.csv")[
    ["wav_path", "orig_dur_s", "trim_dur_s", "peak", "seg_rms_db"]]
df = idx.merge(man, on="uuid", how="left").merge(seg, on="wav_path", how="left")
assert len(df) == len(X_full) == len(logmel) and (df.uuid.values == idx.uuid.values).all()

dt = pd.to_datetime(df.datetime, utc=True, errors="coerce")
df["days"] = ((dt - dt.min()).dt.total_seconds() / 86400).fillna(-1)
df["gender_code"] = df.gender.map({"male": 0, "female": 1, "other": 2}).fillna(-1)
df["age"] = df.age.fillna(-1)
df["SNR"] = df.SNR.fillna(-99)
META = ["orig_dur_s", "trim_dur_s", "peak", "seg_rms_db", "SNR", "cough_detected",
        "age", "gender_code", "days"]

# real (unpadded) content length of every segment
n_valid = np.minimum(df.trim_dur_s.values * SR, WIN)
df["pad_frac"] = 1 - n_valid / WIN
nf = np.clip((n_valid // HOP).astype(int), 9, logmel.shape[2])
print("pad_frac over all clips:\n", df.pad_frac.describe().round(3))
print("clips with >5% padding:", f"{(df.pad_frac > 0.05).mean():.1%}")


def masked_stats(i):
    S = np.asarray(logmel[i][:, :nf[i]])
    m = librosa.feature.mfcc(S=S, n_mfcc=40)
    d = librosa.feature.delta(m, width=9)
    return np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)]).astype(np.float32)


X_mask = np.stack([masked_stats(i) for i in tqdm(range(len(df)), desc="masked MFCC")])


def rf():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1,
                                  class_weight="balanced_subsample", random_state=SEED)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))


def cv_auc(make_model, X, y, g):
    out = []
    for r in range(N_REPEATS):
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + r)
        for tr, te in sgkf.split(np.zeros(len(y)), y, g):
            m = make_model().fit(X[tr], y[tr])
            out.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.array(out)


def report(title, sets, y, g, rows, task, subset):
    print(f"\n{title}")
    for name, X in sets.items():
        for mname, mk in (("RF", rf), ("LogReg", lr)):
            a = cv_auc(mk, X, y, g)
            print(f"   {name:18s} {mname:6s} {a.mean():.3f} +/- {a.std():.3f}")
            rows.append({"task": task, "subset": subset, "set": name, "model": mname,
                         "auc": a.mean(), "std": a.std()})


rows = []
for task, (ok, ycol) in {"expert": ("expert_ok", "y_expert"), "self": ("self_ok", "y_self")}.items():
    sel = np.where(df[ok] & df.split.isin(["train", "val"]))[0]
    y = df.loc[sel, ycol].astype(int).values
    g = df.loc[sel, "group_proxy"].values
    print(f"\n{'=' * 8} TASK: {task} | n={len(sel)} | positives {y.mean():.1%} {'=' * 8}")
    print("mean pad_frac by class (0/1):", df.loc[sel].groupby(ycol).pad_frac.mean().round(3).to_dict())
    print("cough_detected in this set:\n", df.loc[sel, "cough_detected"].describe().round(4))

    sets = {"pad_frac alone": df.loc[sel, ["pad_frac"]].values,
            "audio_full": X_full[sel], "audio_masked": X_mask[sel],
            "meta": df.loc[sel, META].values,
            "audio_masked+meta": np.hstack([X_mask[sel], df.loc[sel, META].values])}
    report("A) All clips:", sets, y, g, rows, task, "all")

    keep = df.pad_frac.values[sel] < 0.02            # clips with essentially no padding
    ys, gs, ss = y[keep], g[keep], sel[keep]
    print(f"\nB) Unpadded subset: n={len(ss)} | class counts {np.bincount(ys, minlength=2).tolist()}")
    if min(np.bincount(ys, minlength=2)) >= 60:
        sets_b = {"audio": X_full[ss], "meta": df.loc[ss, META].values}
        report("   Unpadded subset results:", sets_b, ys, gs, rows, task, "unpadded")
    else:
        print("   too few samples in one class, skipped")

pd.DataFrame(rows).to_csv(ROOT / "results/tables/audit_padding.csv", index=False)
print("\nSaved results/tables/audit_padding.csv")