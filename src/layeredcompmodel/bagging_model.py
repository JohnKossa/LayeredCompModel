import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize_scalar
from typing import Any, List, Optional, Union

from layeredcompmodel.model import LayeredCompModel


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

        metric_fn = mean_absolute_error if self.split_metric == 'mae' else mean_squared_error

        random_state = check_random_state(self.random_state)

        for i in range(self.tree_count):
            seed = random_state.randint(np.iinfo(np.int32).max)
            X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=(1 - self.sample_pct),
                                                      random_state=seed)

            tree = LayeredCompModel(split_metric=self.split_metric, n_jobs=self.n_jobs)
            tree.fit(X_tr, y_tr)

            def objective(w: float) -> float:
                tree.weight_falloff = w
                preds = tree.predict(X_ts)
                return float(metric_fn(y_ts, preds))

            if len(y_ts) > 0:
                res = minimize_scalar(objective, bounds=(0.0, 15.0), method='bounded')
                opt_w = res.x
                best = res.fun
            else:
                # Fallback if no test data
                opt_w = 3
                best = -1

            tree.weight_falloff = opt_w
            self.estimators_.append(tree)
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
