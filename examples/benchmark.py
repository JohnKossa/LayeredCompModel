"""Benchmark LayeredCompModel and LayeredCompBaggingModel against baselines.

Compares accuracy (MAE / MAPE / RMSE) and wall-clock fit/predict time on a
shared synthetic real-estate dataset against:

  * scikit-learn ``LinearRegression`` (one-hot encoded features)
  * XGBoost ``XGBRegressor`` (optional -- skipped with a note if not installed)

The LayeredComp models consume the raw frame (categoricals handled natively);
the baselines receive a one-hot-encoded copy of the same split.

Run::

    python examples/benchmark.py

XGBoost is optional::

    pip install layeredcompmodel[benchmark]   # or: pip install xgboost
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split

from layeredcompmodel import LayeredCompBaggingModel, LayeredCompModel

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_XGBOOST = False

RANDOM_STATE = 42
N_SAMPLES = 2000


def make_dataset(n_samples: int, seed: int) -> pd.DataFrame:
    """Synthetic real-estate-like data with a non-linear price signal.

    The signal is deliberately hard for a plain linear model: price-per-sqft
    varies by neighborhood (interaction), age contributes non-monotonically
    (a "historic charm" bump for pre-1940 homes on top of newer-is-better),
    and there is a step change once a home clears 3000 sqft (luxury regime).
    A hierarchical tree that narrows the comparison group should capture these
    where a single global slope cannot.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "neighborhood": rng.choice(
                ["North", "South", "East", "West"], n_samples
            ),
            "size_sqft": rng.normal(2000, 600, n_samples).clip(500, 6000),
            "year_built": rng.integers(1900, 2023, n_samples),
        }
    )

    # Neighborhood-specific price-per-sqft (interaction, not an additive shift).
    price_per_sqft = df["neighborhood"].map(
        {"North": 420.0, "South": 250.0, "East": 310.0, "West": 200.0}
    )
    price = price_per_sqft * df["size_sqft"]

    # Non-monotonic age effect: newer homes worth more, but pre-1940 homes get
    # a "historic charm" premium that reverses the trend at the old end.
    price += (df["year_built"] - 1900) * 900
    price += np.where(df["year_built"] < 1940, 80000, 0)

    # Luxury regime: a step change once a home clears 3000 sqft.
    price += np.where(df["size_sqft"] > 3000, 150000, 0)

    # Mild curvature so the marginal sqft value tapers off.
    price += -0.02 * (df["size_sqft"] - 2000) ** 2

    price += rng.normal(0, 25000, n_samples)  # additive noise
    price *= rng.normal(1.0, 0.05, n_samples)  # multiplicative noise
    df["price"] = price.clip(lower=50000)
    return df


def time_call(fn: Callable[[], object]) -> tuple[object, float]:
    """Run ``fn`` and return ``(result, elapsed_seconds)``."""
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": 100 * mean_absolute_percentage_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main() -> None:
    df = make_dataset(N_SAMPLES, RANDOM_STATE)
    feature_cols = ["neighborhood", "size_sqft", "year_built"]
    X, y = df[feature_cols], df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # One-hot copy for the numeric-only baselines. align() keeps train/test
    # columns identical if a category is missing from one split.
    X_train_oh = pd.get_dummies(X_train)
    X_test_oh = pd.get_dummies(X_test)
    X_train_oh, X_test_oh = X_train_oh.align(
        X_test_oh, join="left", axis=1, fill_value=0
    )

    results: list[dict[str, object]] = []

    def run_model(name: str, model: object, X_tr: object, X_te: object) -> None:
        _, fit_s = time_call(lambda: model.fit(X_tr, y_train))  # type: ignore[attr-defined]
        preds, pred_s = time_call(lambda: model.predict(X_te))  # type: ignore[attr-defined]
        row: dict[str, object] = {"model": name, "fit_s": fit_s, "predict_s": pred_s}
        row.update(evaluate(y_test, np.asarray(preds)))
        results.append(row)

    run_model(
        "LayeredCompModel",
        LayeredCompModel(weight_falloff=0.8, n_jobs=1),
        X_train,
        X_test,
    )
    run_model(
        "LayeredCompBaggingModel",
        LayeredCompBaggingModel(tree_count=10, sample_pct=0.95, random_state=RANDOM_STATE),
        X_train,
        X_test,
    )
    run_model("LinearRegression", LinearRegression(), X_train_oh, X_test_oh)

    if HAS_XGBOOST:
        run_model(
            "XGBRegressor",
            XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
            ),
            X_train_oh,
            X_test_oh,
        )
    else:
        print("[note] xgboost not installed -- skipping XGBRegressor "
              "(pip install xgboost)\n")

    report = pd.DataFrame(results).set_index("model")
    report = report[["MAE", "MAPE", "RMSE", "fit_s", "predict_s"]]
    pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
    print(f"Benchmark on {len(y_test)} test rows "
          f"({len(y_train)} train, seed={RANDOM_STATE}):\n")
    print(report.to_string())


if __name__ == "__main__":
    main()
