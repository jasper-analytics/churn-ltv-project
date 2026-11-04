from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.home() / "jasper_machine_learning_projects" / "churn-ltv-project"
P = ROOT / "data" / "processed"

df = pd.read_parquet(P / "snapshots.parquet")
df = df[df["split"] != "embargo"].reset_index(drop=True)
df["snap_month"] = df["snapshot"].dt.month

NON_FEATURES = ["Customer ID", "snapshot", "split", "churn_60", "net_revenue_90", "snap_month"]
FEATURES = [c for c in df.columns if c not in NON_FEATURES]
LIFETIME = ["tenure", "orders_total", "revenue_total", "n_days"]   # grow mechanically with snapshot date
NO_LIFE = [c for c in FEATURES if c not in LIFETIME]
WITH_MONTH = FEATURES + ["snap_month"]

tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
print(len(FEATURES), "features | rows:", len(tr), len(va), len(te))


# %% Metrics: pooled AUC plus per-snapshot AUC (what a retention team uses: ranking within one snapshot)
def evaluate(y, p, snaps):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "s": np.asarray(snaps)})
    aucs, precs, lifts = [], [], []
    for _, g in d.groupby("s"):
        aucs.append(roc_auc_score(g["y"], g["p"]))
        top = g.nlargest(max(1, int(round(0.1 * len(g)))), "p")
        precs.append(top["y"].mean())
        lifts.append(top["y"].mean() / g["y"].mean())
    return {"AUC": roc_auc_score(d["y"], d["p"]), "AUC_per_snap": np.mean(aucs),
            "prec_top10": np.mean(precs), "lift_top10": np.mean(lifts)}

results = []
def record(name, split, y, p, snaps, prob=True):
    r = evaluate(y, p, snaps)
    r.update(model=name, split=split)
    if prob:
        r.update(brier=brier_score_loss(y, p), mean_pred=np.mean(p), mean_actual=np.mean(y))
    results.append(r)

for sp, d in (("val", va), ("test", te)):
    record("naive: fewer orders_total", sp, d["churn_60"], -d["orders_total"], d["snapshot"], prob=False)


# %% Logistic baseline (needs imputation, log transforms, clipping, scaling)
SKEWED = ["revenue_30", "revenue_90", "revenue_180", "revenue_prev90", "revenue_total",
          "aov", "avg_items", "return_value_180", "revenue_trend"]
FLAGS = ["is_one_time", "few_gaps"]

def make_logit_matrix(X, ref=None):
    X = X.copy()
    if "avg_gap" in X.columns:
        X["is_one_time"] = X["avg_gap"].isna().astype(int)
    if "gap_std" in X.columns:
        X["few_gaps"] = X["gap_std"].isna().astype(int)
    for c in SKEWED:
        if c in X.columns:
            X[c] = np.sign(X[c]) * np.log1p(X[c].abs())
    if ref is None:   # bounds and medians come from TRAIN only
        ref = {"lo": X.quantile(0.01), "hi": X.quantile(0.99), "med": X.median()}
    X = X.fillna(ref["med"])
    num = [c for c in X.columns if c not in FLAGS]
    X[num] = X[num].clip(lower=ref["lo"][num], upper=ref["hi"][num], axis=1)
    return X, ref

def run_logit(feats, name):
    Xtr, ref = make_logit_matrix(tr[feats])
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
    m.fit(Xtr, tr["churn_60"])
    for sp, d in (("val", va), ("test", te)):
        Xd, _ = make_logit_matrix(d[feats], ref)
        record(name, sp, d["churn_60"], m.predict_proba(Xd)[:, 1], d["snapshot"])
    return m

m_logit = run_logit(FEATURES, "logistic: all features")


# %% LightGBM (early stopping on validation; test is never used for tuning)
def run_lgbm(feats, name):
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=15,
                           min_child_samples=100, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=5.0, random_state=0, verbose=-1)
    m.fit(tr[feats], tr["churn_60"], eval_set=[(va[feats], va["churn_60"])],
          eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
    for sp, d in (("val", va), ("test", te)):
        record(name, sp, d["churn_60"], m.predict_proba(d[feats])[:, 1], d["snapshot"])
    print(f"{name}: best iteration {m.best_iteration_}")
    return m

m_all = run_lgbm(FEATURES, "lgbm: all features")
m_nolife = run_lgbm(NO_LIFE, "lgbm: no lifetime features")
m_month = run_lgbm(WITH_MONTH, "lgbm: all + month")


# %% Results table
cols = ["AUC", "AUC_per_snap", "prec_top10", "lift_top10", "brier", "mean_pred", "mean_actual"]
res = pd.DataFrame(results).set_index(["model", "split"])[cols].round(3)
print(res.sort_index(level="split"))


# %% Calibration (test) and feature importance for the main LightGBM
p = m_all.predict_proba(te[FEATURES])[:, 1]
cal = pd.DataFrame({"p": p, "y": te["churn_60"].values})
cal["bin"] = pd.qcut(cal["p"], 10, duplicates="drop")
print(cal.groupby("bin", observed=True).agg(n=("y", "size"), mean_pred=("p", "mean"), actual=("y", "mean")).round(3))

imp = pd.Series(m_all.booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
print((imp / imp.sum()).head(15).round(3))