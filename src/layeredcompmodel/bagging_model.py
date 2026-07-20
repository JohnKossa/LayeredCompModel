import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize_scalar
from joblib import Parallel, delayed
from typing import Any, List, Optional, Union

from layeredcompmodel.model import LayeredCompModel


def _fit_single_tree(X: Any, y: Any, seed: int, sample_pct: float, split_metric: str):
    """Fit one bagging tree and optimize its weight_falloff on the held-out fold.

    Module-level (not a closure) so it is picklable for joblib's process backend. Fits with
    ``n_jobs=1``: parallelism is across trees, so nesting within-tree parallelism would only
    oversubscribe. Deterministic given ``seed`` — the split, tree build, and falloff search are all
    seeded/deterministic, so the parallel result is bit-for-bit identical to a serial fit.
    """
    metric_fn = mean_absolute_error if split_metric == 'mae' else mean_squared_error
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=(1 - sample_pct), random_state=seed)

    tree = LayeredCompModel(split_metric=split_metric, n_jobs=1)
    tree.fit(X_tr, y_tr)

    if len(y_ts) > 0:
        def objective(w: float) -> float:
            tree.weight_falloff = w
            return float(metric_fn(y_ts, tree.predict(X_ts)))

        res = minimize_scalar(objective, bounds=(0.0, 15.0), method='bounded')
        tree.weight_falloff = res.x
        best = res.fun
    else:
        tree.weight_falloff = 3      # fallback if no held-out data
        best = -1
    return tree, float(best)


class LayeredCompBaggingModel(BaseEstimator, RegressorMixin):
    """
    Layered Comp Bagging Model.

    A bagging ensemble version of the primary algorithm that reduces variance
    and automatically optimizes the weight_falloff for each tree in the ensemble.

    Parameters
    ----------
    tree_count : int, default=10
        Number of trees to build. Must be >= 1.
    sample_pct : float, default=0.8
        Fraction of data sampled for each tree and used as the internal split ratio.
        Must be between 0 and 1 (exclusive).
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for subsampling.
    split_metric : {'mae', 'mse'}, default='mae'
        Metric used for both tree splitting and weight_falloff optimization.
    """

    def __init__(
            self,
            tree_count: int = 10,
            sample_pct: float = 0.8,
            random_state: Optional[Union[int, np.random.RandomState]] = None,
            split_metric: str = 'mae',
            n_jobs: int = 1
    ) -> None:
        self.tree_count = tree_count
        self.sample_pct = sample_pct
        self.random_state = random_state
        self.split_metric = split_metric
        self.n_jobs = n_jobs

    def fit(self, X: Any, y: Any) -> "LayeredCompBaggingModel":
        """
        Build a bagging ensemble of LayeredCompModel trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate hyperparameters
        if self.tree_count < 1:
            raise ValueError(f"tree_count must be >= 1, got {self.tree_count}")
        if not (0 < self.sample_pct < 1):
            raise ValueError(f"sample_pct must be between 0 and 1 (exclusive), got {self.sample_pct}")
        if self.split_metric not in ('mae', 'mse'):
            raise ValueError(f"split_metric must be 'mae' or 'mse', got {self.split_metric}")

        if X.shape[1] == 0:
            raise ValueError(f"0 feature(s) (shape={X.shape}) while a minimum of 1 is required.")
        if len(X) == 0:
            raise ValueError(f"Found array with 0 sample(s) (shape={X.shape}) while a minimum of 1 is required.")
        if len(y) == 0:
            raise ValueError(f"Found array with 0 sample(s) (shape={y.shape}) while a minimum of 1 is required.")

        # Convert y to a common format or handle both types
        y_array = y.values if hasattr(y, 'values') else y

        if pd.isna(y_array).any():
            raise ValueError("Input y contains NaN.")
        if pd.api.types.is_numeric_dtype(y_array) and np.isinf(y_array).any():
            raise ValueError("Input y contains infinity.")

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = getattr(X, "columns", np.array([str(i) for i in range(X.shape[1])])).tolist()

        self.estimators_: List[LayeredCompModel] = []

        random_state = check_random_state(self.random_state)

        # Draw every per-tree seed up front, in order, so the RNG sequence is identical to the old
        # serial loop (=> bit-for-bit determinism). The trees are independent, so fitting them in
        # parallel across ``n_jobs`` processes speeds up the dominant tree-build cost WITHOUT changing
        # any output. Each tree fits with n_jobs=1 (parallelism is across trees, not within).
        seeds = [int(random_state.randint(np.iinfo(np.int32).max)) for _ in range(self.tree_count)]

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_single_tree)(X, y, seed, self.sample_pct, self.split_metric)
            for seed in seeds
        )

        self.estimators_ = [tree for tree, _best in results]
        for i, (tree, best) in enumerate(results):
            print(f"Trained tree {i + 1} of {self.tree_count} with weight {tree.weight_falloff} @ {best}")

        return self

    def predict(self, X: Any) -> np.ndarray:
        """
        Predict regression target for X.

        The final prediction is the arithmetic mean of all individual tree predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)

        all_preds = []
        for tree in self.estimators_:
            all_preds.append(tree.predict(X))

        return np.mean(all_preds, axis=0)

    # ------------------------------------------------------------------ serialization
    SERIAL_FORMAT_VERSION = 1

    def to_dict(self) -> dict:
        """Serialize the fitted ensemble to a JSON-compatible dict (a portable, inspectable,
        pickle-free alternative to persisting the estimator). Round-trips through :meth:`from_dict`
        to a PREDICT-READY model whose predictions match this one bit-for-bit.

        Each tree carries its own structure (``LayeredCompModel.to_dict``) plus the metadata predict
        needs; numeric thresholds serialize as floats and survive JSON round-trip losslessly.
        """
        check_is_fitted(self)
        import layeredcompmodel
        return {
            "format_version": self.SERIAL_FORMAT_VERSION,
            "lib_version": getattr(layeredcompmodel, "__version__", None),
            "split_metric": self.split_metric,
            "n_features_in": int(self.n_features_in_),
            "feature_names": list(self.feature_names_in_),
            "trees": [
                {
                    "weight_falloff": float(t.weight_falloff),
                    "split_metric": t._split_metric_name,
                    "columns": list(t.columns_),
                    "n_features_in": int(t.n_features_in_),
                    "tree": t.to_dict(),
                }
                for t in self.estimators_
            ],
        }

    @classmethod
    def from_dict(cls, state: dict) -> "LayeredCompBaggingModel":
        """Reconstruct a PREDICT-READY ensemble from :meth:`to_dict` output. See
        :meth:`LayeredCompModel.from_dict` for the predict-only-vs-trainable contract."""
        fmt = state.get("format_version")
        if fmt != cls.SERIAL_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported serialization format_version={fmt} "
                f"(this build reads {cls.SERIAL_FORMAT_VERSION})."
            )
        model = cls(tree_count=len(state["trees"]), split_metric=state["split_metric"])
        model.n_features_in_ = int(state["n_features_in"])
        model.feature_names_in_ = list(state["feature_names"])
        model.estimators_ = [LayeredCompModel.from_dict(t) for t in state["trees"]]
        return model

    def to_json(self, indent: int = 2) -> str:
        """Serialize the fitted ensemble to a JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "LayeredCompBaggingModel":
        """Reconstruct a predict-ready ensemble from a JSON string produced by :meth:`to_json`."""
        import json
        return cls.from_dict(json.loads(s))
