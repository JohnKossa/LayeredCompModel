"""Shared deterministic fixture for the perf-equivalence tests.

Exercises every prediction path: numeric splits, categorical one-vs-rest, NaN early-stop (numeric),
NaN category, and unseen test categories (forces the routed-to-missing-child stop). Kept in one place
so the baseline-capture and the equivalence test build byte-identical inputs.
"""
import numpy as np
import pandas as pd

SEED = 20260705


def make_fixture(n_train=400, n_test=600):
    rng = np.random.RandomState(SEED)

    def build(n):
        df = pd.DataFrame({
            "num_a": rng.normal(100, 20, n),
            "num_b": rng.normal(0, 1, n),
            "num_c": rng.uniform(0, 500, n),
            "cat_x": rng.choice(["p", "q", "r", "s"], n).astype(object),
            "cat_y": rng.choice(["y0", "y1", "y2"], n).astype(object),
        })
        df.loc[rng.rand(n) < 0.05, "num_a"] = np.nan       # numeric NaN -> early-stop at fit/predict
        df.loc[rng.rand(n) < 0.05, "cat_x"] = np.nan        # NaN as its own category
        return df

    X = build(n_train)
    X_test = build(n_test)
    X_test.loc[rng.rand(n_test) < 0.03, "cat_y"] = "UNSEEN"  # unseen cat -> routed-to-missing-child
    base = X["num_a"].fillna(X["num_a"].mean())
    y = pd.Series(50_000 + 900 * base + 8_000 * X["num_b"] + rng.normal(0, 5_000, n_train))
    return X, X_test, y
