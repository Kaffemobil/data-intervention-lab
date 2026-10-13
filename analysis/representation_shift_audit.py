"""Audit whether the learned VQ representation exposes temporal distribution shift.

Fits the same 32-code MiniBatchKMeans codebook on pre-2025 standardized
features only, assigns the 2025+ holdout to the frozen codebook, and reports:
- active/effective state counts
- train-vs-holdout Jensen-Shannon divergence
- state support vs state-conditioned holdout error
- paired practical effect size for 25% VQ-balanced vs random sampling

This is an exploratory representation audit; it does not alter the primary
forecasting stage gate.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "vq_modeling_table.csv")
fixed = ["hs", "origin_period", "m1_error"]
features = [c for c in df.columns if c not in fixed]
train = df[df.origin_period < "2025-01"].copy()
hold = df[df.origin_period >= "2025-01"].copy()

sc = StandardScaler().fit(train[features])
Xtr = sc.transform(train[features])
Xho = sc.transform(hold[features])
ytr = train.m1_error.to_numpy()
yho = hold.m1_error.to_numpy()

dev = train.origin_period < "2024-01"
val = train.origin_period >= "2024-01"
alphas = [0.1, 1, 10, 100, 1000, 10000]
alpha = min(
    (
        (a, mean_squared_error(
            ytr[val],
            Ridge(alpha=a).fit(Xtr[dev], ytr[dev]).predict(Xtr[val])
        ) ** 0.5)
        for a in alphas
    ),
    key=lambda x: x[1],
)[0]

K = 32
km = MiniBatchKMeans(
    n_clusters=K, random_state=7, n_init=20, batch_size=256
).fit(Xtr)
ctr = km.labels_
cho = km.predict(Xho)
ntr = np.bincount(ctr, minlength=K)
nho = np.bincount(cho, minlength=K)
p = ntr / ntr.sum()
q = nho / nho.sum()

def eff(prob):
    x = prob[prob > 0]
    return float(np.exp(-np.sum(x * np.log(x))))

def js_bits(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        z = a > 0
        return float(np.sum(a[z] * np.log2(a[z] / b[z])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

model = Ridge(alpha=alpha).fit(Xtr, ytr)
err2 = (yho - model.predict(Xho)) ** 2
rows = []
for k in range(K):
    mask = cho == k
    rows.append({
        "code": k,
        "train_n": int(ntr[k]),
        "holdout_n": int(nho[k]),
        "train_share": float(p[k]),
        "holdout_share": float(q[k]),
        "holdout_rmse": float(np.sqrt(np.mean(err2[mask]))) if mask.any() else np.nan,
    })
states = pd.DataFrame(rows)
active = states[states.holdout_n > 0].copy()
rho = float(np.corrcoef(
    active.train_n.rank().to_numpy(),
    active.holdout_rmse.rank().to_numpy()
)[0,1])

rep = pd.read_csv(ROOT / "data" / "vq_sampling_replicates.csv")
pair = rep[rep.budget == .25].pivot(index="seed", columns="method", values="test_rmse")
delta = (pair["VQ-balanced"] - pair["Random"]).to_numpy()
rng = np.random.default_rng(20261013)
boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(10000)])

summary = {
    "train_effective_states": eff(p),
    "holdout_active_codes": int((nho > 0).sum()),
    "holdout_effective_states": eff(q),
    "js_divergence_bits": js_bits(p, q),
    "support_vs_holdout_rmse_spearman": rho,
    "vq25_mean_delta_rmse_vs_random": float(delta.mean()),
    "vq25_bootstrap_mean_delta_ci95": [float(x) for x in np.quantile(boot, [.025,.975])],
    "vq25_fraction_seeds_better": float((delta < 0).mean()),
}
print(json.dumps(summary, indent=2))
