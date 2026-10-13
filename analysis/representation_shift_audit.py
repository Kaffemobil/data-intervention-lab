"""Audit temporal distribution shift in the frozen 32-code representation.

The same codebook used in the sampling experiment is fitted on pre-2025 data
only and then applied without refitting to the 2025+ holdout. This is an
exploratory diagnostic and does not alter the primary forecasting stage gate.

Outputs
-------
data/representation_shift_states.csv
data/representation_shift_summary.json
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
K = 32


def effective_states(prob):
    x = prob[prob > 0]
    return float(np.exp(-np.sum(x * np.log(x))))


def js_bits(p, q):
    midpoint = 0.5 * (p + q)

    def kl(a, b):
        keep = a > 0
        return float(np.sum(a[keep] * np.log2(a[keep] / b[keep])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def main():
    df = pd.read_csv(DATA / "vq_modeling_table.csv")
    fixed = ["hs", "origin_period", "m1_error"]
    features = [c for c in df.columns if c not in fixed]
    train = df[df.origin_period < "2025-01"].copy()
    holdout = df[df.origin_period >= "2025-01"].copy()

    scaler = StandardScaler().fit(train[features])
    X_train = scaler.transform(train[features])
    X_holdout = scaler.transform(holdout[features])
    y_train = train.m1_error.to_numpy()
    y_holdout = holdout.m1_error.to_numpy()

    dev = (train.origin_period < "2024-01").to_numpy()
    val = (train.origin_period >= "2024-01").to_numpy()
    alphas = [0.1, 1, 10, 100, 1000, 10000]
    alpha = min(
        (
            (
                a,
                mean_squared_error(
                    y_train[val],
                    Ridge(alpha=a)
                    .fit(X_train[dev], y_train[dev])
                    .predict(X_train[val]),
                )
                ** 0.5,
            )
            for a in alphas
        ),
        key=lambda x: x[1],
    )[0]

    codebook = MiniBatchKMeans(
        n_clusters=K, random_state=7, n_init=20, batch_size=256
    ).fit(X_train)
    train_codes = codebook.labels_
    holdout_codes = codebook.predict(X_holdout)

    n_train = np.bincount(train_codes, minlength=K)
    n_holdout = np.bincount(holdout_codes, minlength=K)
    p = n_train / n_train.sum()
    q = n_holdout / n_holdout.sum()

    model = Ridge(alpha=alpha).fit(X_train, y_train)
    err2 = (y_holdout - model.predict(X_holdout)) ** 2

    rows = []
    for code in range(K):
        mask = holdout_codes == code
        rows.append(
            {
                "code": code,
                "train_n": int(n_train[code]),
                "holdout_n": int(n_holdout[code]),
                "train_share": float(p[code]),
                "holdout_share": float(q[code]),
                "holdout_rmse": (
                    float(np.sqrt(np.mean(err2[mask]))) if mask.any() else np.nan
                ),
            }
        )
    states = pd.DataFrame(rows)
    states.to_csv(DATA / "representation_shift_states.csv", index=False)

    active = states[states.holdout_n > 0].copy()
    rho = float(
        np.corrcoef(
            active.train_n.rank().to_numpy(),
            active.holdout_rmse.rank().to_numpy(),
        )[0, 1]
    )

    replicates = pd.read_csv(DATA / "vq_sampling_replicates.csv")
    paired = replicates[replicates.budget == 0.25].pivot(
        index="seed", columns="method", values="test_rmse"
    )
    delta = (paired["VQ-balanced"] - paired["Random"]).to_numpy()
    rng = np.random.default_rng(20261013)
    bootstrap = np.array(
        [rng.choice(delta, len(delta), replace=True).mean() for _ in range(10000)]
    )

    summary = {
        "train_effective_states": effective_states(p),
        "holdout_active_codes": int((n_holdout > 0).sum()),
        "holdout_effective_states": effective_states(q),
        "js_divergence_bits": js_bits(p, q),
        "support_vs_holdout_rmse_spearman": rho,
        "vq25_mean_delta_rmse_vs_random": float(delta.mean()),
        "vq25_bootstrap_mean_delta_ci95": [
            float(x) for x in np.quantile(bootstrap, [0.025, 0.975])
        ],
        "vq25_fraction_seeds_better": float((delta < 0).mean()),
    }
    (DATA / "representation_shift_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
