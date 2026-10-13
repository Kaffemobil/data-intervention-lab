"""Reproduce the 32-code vector-quantized sampling experiment.

Input
-----
data/vq_modeling_table.csv

Protocol
--------
- Train: origin_period < 2025-01
- Future holdout: origin_period >= 2025-01
- Representation: MiniBatchKMeans(k=32), fit on standardized training features only
- Selection: uniform random vs VQ-balanced (one example per occupied code first,
  then sqrt-size allocation across codes)
- Downstream model: Ridge predicting the frozen baseline's 3-month residual
- Replicates: 200 seeds at 25%, 50%, and 75% training-data budgets

Outputs
-------
data/vq_sampling_replicates.csv
data/vq_sampling_summary.csv
data/codebook_summary.csv

The script is the source of truth for the public VQ numbers. It intentionally
writes the derived tables used by the site so the checked-in outputs can be
regenerated rather than merely inspected.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
K = 32
BUDGETS = (0.25, 0.50, 0.75)
SEEDS = range(200)


def select_alpha(X, y, periods):
    """Tune Ridge alpha on a strictly earlier slice, then freeze it."""
    dev = periods < "2024-01"
    val = periods >= "2024-01"
    alphas = [0.1, 1, 10, 100, 1000, 10000]
    scores = []
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(X[dev], y[dev])
        rmse = mean_squared_error(y[val], model.predict(X[val])) ** 0.5
        scores.append((alpha, rmse))
    return min(scores, key=lambda x: x[1])[0]


def vq_sample(codes, n, rng):
    """Coverage-first sample, then allocate remaining draws by sqrt(code size)."""
    clusters = [c for c in range(K) if np.any(codes == c)]
    chosen = []

    if n >= len(clusters):
        for c in clusters:
            members = np.where(codes == c)[0]
            chosen.append(int(rng.choice(members)))
        remaining = n - len(chosen)
    else:
        selected_codes = rng.choice(clusters, size=n, replace=False)
        for c in selected_codes:
            members = np.where(codes == c)[0]
            chosen.append(int(rng.choice(members)))
        return np.asarray(chosen, dtype=int)

    if remaining:
        chosen_set = set(chosen)
        available = {
            c: [int(i) for i in np.where(codes == c)[0] if int(i) not in chosen_set]
            for c in clusters
        }
        base_weights = np.array(
            [np.sqrt(len(available[c]) + 1e-9) for c in clusters], dtype=float
        )
        for _ in range(remaining):
            valid = np.array([len(available[c]) > 0 for c in clusters])
            if not valid.any():
                break
            weights = base_weights * valid
            weights /= weights.sum()
            c = int(rng.choice(clusters, p=weights))
            j = int(rng.integers(len(available[c])))
            chosen.append(available[c].pop(j))

    return np.asarray(chosen, dtype=int)


def tv_distance(sample_codes, full_code_probs):
    sample_probs = np.bincount(sample_codes, minlength=K) / len(sample_codes)
    return 0.5 * np.abs(sample_probs - full_code_probs).sum()


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

    alpha = select_alpha(X_train, y_train, train.origin_period.to_numpy())

    codebook = MiniBatchKMeans(
        n_clusters=K,
        random_state=7,
        n_init=20,
        batch_size=256,
    ).fit(X_train)
    codes = codebook.labels_
    full_code_probs = np.bincount(codes, minlength=K) / len(codes)

    records = []
    for frac in BUDGETS:
        n = round(frac * len(train))
        for seed in SEEDS:
            # One RNG stream per replicate; method order is frozen by protocol.
            rng = np.random.default_rng(seed)
            idx_random = rng.choice(len(train), n, replace=False)
            idx_vq = vq_sample(codes, n, rng)

            for method, idx in (("Random", idx_random), ("VQ-balanced", idx_vq)):
                model = Ridge(alpha=alpha).fit(X_train[idx], y_train[idx])
                pred = model.predict(X_holdout)
                records.append(
                    {
                        "budget": frac,
                        "seed": seed,
                        "method": method,
                        "n_train": n,
                        "test_rmse": mean_squared_error(y_holdout, pred) ** 0.5,
                        "code_coverage": len(np.unique(codes[idx])) / K,
                        "code_tv_distance": tv_distance(codes[idx], full_code_probs),
                    }
                )

    replicates = pd.DataFrame(records)
    replicates.to_csv(DATA / "vq_sampling_replicates.csv", index=False)

    summary = (
        replicates.groupby(["budget", "method"])
        .agg(
            rmse_mean=("test_rmse", "mean"),
            rmse_sd=("test_rmse", "std"),
            rmse_p05=("test_rmse", lambda x: np.quantile(x, 0.05)),
            rmse_p95=("test_rmse", lambda x: np.quantile(x, 0.95)),
            coverage_mean=("code_coverage", "mean"),
            coverage_min=("code_coverage", "min"),
            tv_mean=("code_tv_distance", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(DATA / "vq_sampling_summary.csv", index=False)

    code_rows = []
    for code in range(K):
        center = codebook.cluster_centers_[code]
        top = np.argsort(np.abs(center))[::-1][:5]
        top_features = " | ".join(
            f"{features[j]} ({center[j]:+.2f}σ)" for j in top
        )
        code_rows.append(
            {
                "code": code,
                "n_train": int((codes == code).sum()),
                "top_features": top_features,
            }
        )
    pd.DataFrame(code_rows).to_csv(DATA / "codebook_summary.csv", index=False)

    full_model = Ridge(alpha=alpha).fit(X_train, y_train)
    full_rmse = mean_squared_error(y_holdout, full_model.predict(X_holdout)) ** 0.5

    print("selected alpha:", alpha)
    print("train rows:", len(train), "holdout rows:", len(holdout))
    print("full-data holdout RMSE:", f"{full_rmse:.6f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
