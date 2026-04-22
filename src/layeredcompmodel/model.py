import numpy as np
import pandas as pd
import json
from sklearn.base import BaseEstimator, RegressorMixin

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from joblib import Parallel, delayed
from collections import deque

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pandas import DataFrame, Series


def calculate_wilson_mean(y: Union[np.ndarray, pd.Series, List[float]]) -> float:
    """
    Calculates the Wilson mean: trim the top 2.5% and the bottom 2.5% (the middle 95%)
    and return the mean of the remaining data.
    """
    if len(y) == 0:
        return 0.0
    if len(y) < 40:  # 1/0.025 = 40. For small samples, trimming might remove everything or too much.
        # However, the spec says "trim top 2.5% and bottom 2.5%".
        # For small N, we should ensure at least some data remains if possible,
        # or follow the standard np.percentile approach.
        pass

    low = np.percentile(y, 2.5)
    high = np.percentile(y, 97.5)
    trimmed_y = y[(y >= low) & (y <= high)]

    if len(trimmed_y) == 0:
        return np.mean(y)
    return np.mean(trimmed_y)


class CompNode:
    def __init__(self, depth: int, wilson_mean: float, count: int, filter_col: Optional[str] = None, filter_val: Optional[Union[str, float]] = None, is_numeric: bool = False, variant: Optional[str] = None) -> None:
        self.depth = depth
        self.wilson_mean = wilson_mean
        self.count = count
        self.filter_col = filter_col
        self.filter_val = filter_val
        self.is_numeric = is_numeric
        self.variant = variant
        self.children: List[CompNode] = []


class LayeredCompModel(BaseEstimator, RegressorMixin):
    def __init__(self, weight_falloff: float = 0.5, split_metric: str = 'mae', n_jobs: int = 1) -> None:
        self.weight_falloff: float = weight_falloff
        self.split_metric_name: str = split_metric
        self.n_jobs: int = n_jobs

        if split_metric == 'mae':
            self.split_metric: Callable[[np.ndarray], float] = self._get_mae
        elif split_metric == 'mse':
            self.split_metric: Callable[[np.ndarray], float] = self._get_mse
        else:
            raise ValueError(f"Invalid split_metric: {split_metric}. Supported metrics are 'mae' and 'mse'.")

        self.tree_: Optional[CompNode] = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return {
            "weight_falloff": self.weight_falloff,
            "split_metric": self.split_metric_name,
            "n_jobs": self.n_jobs,
        }

    def set_params(self, **params: Any) -> "LayeredCompModel":
        for key, value in params.items():
            if key == "split_metric":
                self.split_metric_name: str = value
                if value == 'mae':
                    self.split_metric: Callable[[np.ndarray], float] = self._get_mae
                elif value == 'mse':
                    self.split_metric: Callable[[np.ndarray], float] = self._get_mse
                else:
                    raise ValueError(f"Invalid split_metric: {value}. Supported metrics are 'mae' and 'mse'.")
            else:
                setattr(self, key, value)
        return self

    def _get_mae(self, y_subset: np.ndarray) -> float:
        if len(y_subset) < 2:
            return np.inf

        # y_subset is now a numpy array
        mean = np.mean(y_subset)

        mae = np.mean(np.abs(y_subset - mean))
        return mae

    def _get_mse(self, y_subset: np.ndarray) -> float:
        if len(y_subset) < 2:
            return np.inf

        # y_subset is now a numpy array
        mean = np.mean(y_subset)

        mse = np.mean((y_subset - mean) ** 2)
        return mse

    def _find_best_split(self, X_full: DataFrame, y_full: Series, indices: np.ndarray, columns: List[str], pre_sorted_indices: Optional[Dict[str, np.ndarray]] = None) -> Optional[Tuple[str, Union[str, float], bool]]:
        # We want to MINIMIZE the weighted MAE / base MAE ratio
        # Initializing best_score with 1.0 (no improvement)
        best_score = 1.0
        best_split = None  # (col, val, is_numeric)

        y = y_full.iloc[indices]
        # X = X_full.iloc[indices] # Removed since we use indices for filtering

        total_count = len(y)
        y_values = y.values
        base_metric = self.split_metric(y_values)

        if base_metric == 0 or base_metric == np.inf:
            # Cannot improve or not enough data
            return None

        # Create a set for O(1) membership check during filtering
        indices_set = set(indices)

        for col in columns:
            is_numeric = pd.api.types.is_numeric_dtype(X_full[col])

            if is_numeric:
                # Optimized Numeric split logic using pre-sorted maps
                if pre_sorted_indices and col in pre_sorted_indices:
                    # Filter pre-sorted indices to keep only those present in the current node
                    # Optimized filtering using np.isin
                    col_pre_sorted = pre_sorted_indices[col]
                    mask_in_node = np.isin(col_pre_sorted, indices)
                    col_sorted_indices = col_pre_sorted[mask_in_node]

                    if len(col_sorted_indices) < 8:
                        continue

                    y_values_all = y_full.values
                    X_col_values_all = X_full[col].values

                    y_col_values = y_values_all[col_sorted_indices]
                    X_col_values = X_col_values_all[col_sorted_indices]

                    # Pre-calculate prefix sums/counts for faster metric updates if possible
                    # But self.split_metric (MAE/MSE) depends on the MEAN of the subset, which changes.
                    # So we still need to call the metric function.

                    best_col_score = 1.0
                    best_col_midpoint = None

                    # Binary search on indices of the sorted array
                    num_iterations = min(10, int(np.log2(len(col_sorted_indices))))
                    low_idx = 0
                    high_idx = len(col_sorted_indices) - 1

                    for _ in range(num_iterations):
                        mid_idx = (low_idx + high_idx) // 2

                        # Ensure we split at a point where the value changes to avoid redundant checks
                        # and ensure consistency with <= logic.
                        # If X_col_values[mid_idx] == X_col_values[mid_idx+1], we should move mid_idx
                        curr_mid_idx = mid_idx
                        while curr_mid_idx < high_idx and X_col_values[curr_mid_idx] == X_col_values[curr_mid_idx + 1]:
                            curr_mid_idx += 1

                        if curr_mid_idx == high_idx:
                            # Try searching backwards
                            curr_mid_idx = mid_idx
                            while curr_mid_idx > low_idx and X_col_values[curr_mid_idx] == X_col_values[
                                curr_mid_idx - 1]:
                                curr_mid_idx -= 1
                            if curr_mid_idx == low_idx:
                                # All values in this range are the same
                                high_idx = mid_idx - 1
                                continue
                            else:
                                curr_mid_idx -= 1  # Split point is BEFORE this value

                        midpoint = X_col_values[curr_mid_idx]

                        y_low = y_col_values[:curr_mid_idx + 1]
                        y_high = y_col_values[curr_mid_idx + 1:]

                        metric_low = self.split_metric(y_low)
                        metric_high = self.split_metric(y_high)

                        weighted_metric = (metric_low * len(y_low) + metric_high * len(y_high)) / len(y_col_values)
                        score = weighted_metric / base_metric

                        if score < best_col_score:
                            best_col_score = score
                            best_col_midpoint = midpoint
                        elif score == best_col_score:
                            # Tie break: split most evenly
                            if abs(len(y_low) - len(y_col_values) / 2) < abs(
                                    (len(y_col_values) - len(y_low)) - len(y_col_values) / 2):
                                best_col_midpoint = midpoint

                        if metric_low < metric_high:
                            high_idx = mid_idx - 1
                        else:
                            low_idx = mid_idx + 1

                    if best_col_score < best_score:
                        best_score = best_col_score
                        best_split = (col, best_col_midpoint, True)
                else:
                    # Fallback to old numeric split logic (if no pre_sorted_indices)
                    X_col_full = X_full[col].iloc[indices]
                    mask = X_col_full.notna()
                    y_col = y.iloc[mask.values].values
                    X_col_values_ser = X_col_full.iloc[mask.values]

                    if len(y_col) < 8:
                        continue

                    num_iterations = min(10, int(np.log2(len(y_col))))
                    best_col_score = 1.0
                    best_col_midpoint = None

                    feature_values = X_col_values_ser.sort_values()
                    low_idx = 0
                    high_idx = len(feature_values) - 1

                    for _ in range(num_iterations):
                        mid_idx = (low_idx + high_idx) // 2
                        midpoint = feature_values.iloc[mid_idx]

                        lower_mask = X_col_values_ser <= midpoint

                        y_low = y_col[lower_mask.values]
                        y_high = y_col[~lower_mask.values]

                        if len(y_low) == 0 or len(y_high) == 0:
                            if len(y_low) == 0:
                                low_idx = mid_idx + 1
                            else:
                                high_idx = mid_idx - 1
                            continue

                        metric_low = self.split_metric(y_low)
                        metric_high = self.split_metric(y_high)

                        weighted_metric = (metric_low * len(y_low) + metric_high * len(y_high)) / len(y_col)
                        score = weighted_metric / base_metric

                        if score < best_col_score:
                            best_col_score = score
                            best_col_midpoint = midpoint
                        elif score == best_col_score:
                            current_best_low_count = (
                                        X_col_values_ser <= best_col_midpoint).sum() if best_col_midpoint is not None else 0
                            if abs(len(y_low) - len(y_col) / 2) < abs(current_best_low_count - len(y_col) / 2):
                                best_col_midpoint = midpoint

                        if metric_low < metric_high:
                            high_idx = mid_idx - 1
                        else:
                            low_idx = mid_idx + 1

                    if best_col_score < best_score:
                        best_score = best_col_score
                        best_split = (col, best_col_midpoint, True)
            else:
                # Categorical split logic (one-vs-rest)
                # Treat NaNs as a distinct category
                X_col_filled = X_full[col].iloc[indices].fillna("NaN")
                variants = X_col_filled.unique()
                X_col_filled_values = X_col_filled.values

                for var in variants:
                    mask = X_col_filled_values == var
                    y_v = y_values[mask]
                    y_inv = y_values[~mask]

                    if len(y_v) == 0 or len(y_inv) == 0:
                        continue

                    metric_v = self.split_metric(y_v)
                    metric_inv = self.split_metric(y_inv)

                    weighted_metric = (metric_v * len(y_v) + metric_inv * len(y_inv)) / total_count
                    score = weighted_metric / base_metric

                    if score < best_score:
                        best_score = score
                        best_split = (col, var, False)
                    elif score == best_score:
                        # Tie break
                        current_best_v_count = (X_col_filled_values == best_split[1]).sum() if best_split and not \
                        best_split[2] else 0
                        if abs(len(y_v) - total_count / 2) < abs(current_best_v_count - total_count / 2):
                            best_split = (col, var, False)

        return best_split

    def fit(self, X: DataFrame, y: Series, verbose: bool = False) -> "LayeredCompModel":
        # Convert to pandas for easier manipulation
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        # Ensure it's a dataframe if it was already something else (like a list)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if len(X) == 0 or len(y) == 0 or X.shape[1] == 0:
            raise ValueError("Found input data with 0 samples or 0 features.")
        self.columns_: List[str] = X.columns.tolist()

        # Pre-calculate sorted index maps for numeric columns
        self.pre_sorted_indices_: Dict[str, np.ndarray] = {}
        for col in self.columns_:
            if pd.api.types.is_numeric_dtype(X[col]):
                # Drop NaNs and sort
                valid_mask = X[col].notna()
                # Use integer positions (0..N-1) for indexing into .values later
                valid_indices = np.where(valid_mask)[0]
                sorted_local_idx = np.argsort(X[col].values[valid_mask])
                self.pre_sorted_indices_[col] = valid_indices[sorted_local_idx]

        # Use integer positions for internal processing
        indices = np.arange(len(X))
        self.tree_ = self._build_tree(X, y, indices, depth=0, verbose=verbose)
        return self

    def _build_tree(self, X_full: DataFrame, y_full: Series, indices: np.ndarray, depth: int, variant: Optional[str] = None, verbose: bool = False) -> CompNode:
        y_initial = y_full.iloc[indices]
        root_node_mean = calculate_wilson_mean(y_initial)
        root_node = CompNode(depth=depth, wilson_mean=root_node_mean, count=len(y_initial), variant=variant)

        # queue stores (node, indices)
        queue = deque([(root_node, indices)])

        while queue:
            # If n_jobs > 1, we could process the current level of the queue in parallel.
            # However, a simple BFS with joblib.Parallel over the current queue is more efficient.

            nodes_to_process = []
            while queue:
                nodes_to_process.append(queue.popleft())

            if not nodes_to_process:
                break

            # Prepare tasks for find_best_split
            # We only process nodes that can be split
            tasks = []
            valid_nodes_indices = []
            for i, (node, node_indices) in enumerate(nodes_to_process):
                if len(node_indices) >= 2:
                    tasks.append(delayed(self._find_best_split)(X_full, y_full, node_indices, self.columns_,
                                                                self.pre_sorted_indices_))
                    valid_nodes_indices.append(i)

            if tasks:
                results = Parallel(n_jobs=self.n_jobs)(tasks)

                for idx, split in zip(valid_nodes_indices, results):
                    if split:
                        node, node_indices = nodes_to_process[idx]
                        col, val, is_numeric = split
                        node.filter_col = col
                        node.filter_val = val
                        node.is_numeric = is_numeric

                        X_s = X_full.iloc[node_indices]

                        if is_numeric:
                            mask_notna = X_s[col].notna()
                            # indices are already integer positions
                            valid_sub_indices = node_indices[mask_notna.values]
                            X_clean_values = X_full[col].values[valid_sub_indices]

                            mask_low = X_clean_values <= val
                            indices_low = valid_sub_indices[mask_low]
                            indices_high = valid_sub_indices[~mask_low]

                            if len(indices_low) > 0:
                                y_low = y_full.iloc[indices_low]
                                child_low = CompNode(depth=node.depth + 1,
                                                     wilson_mean=calculate_wilson_mean(y_low),
                                                     count=len(y_low),
                                                     variant='<=')
                                node.children.append(child_low)
                                queue.append((child_low, indices_low))
                            if len(indices_high) > 0:
                                y_high = y_full.iloc[indices_high]
                                child_high = CompNode(depth=node.depth + 1,
                                                      wilson_mean=calculate_wilson_mean(y_high),
                                                      count=len(y_high),
                                                      variant='>')
                                node.children.append(child_high)
                                queue.append((child_high, indices_high))
                        else:
                            X_col = X_s[col].fillna("NaN")
                            mask = (X_col == val).values
                            indices_v = node_indices[mask]
                            indices_rest = node_indices[~mask]

                            if len(indices_v) > 0:
                                y_v = y_full.iloc[indices_v]
                                child_v = CompNode(depth=node.depth + 1,
                                                   wilson_mean=calculate_wilson_mean(y_v),
                                                   count=len(y_v),
                                                   variant='=')
                                node.children.append(child_v)
                                queue.append((child_v, indices_v))
                            if len(indices_rest) > 0:
                                y_rest = y_full.iloc[indices_rest]
                                child_rest = CompNode(depth=node.depth + 1,
                                                      wilson_mean=calculate_wilson_mean(y_rest),
                                                      count=len(y_rest),
                                                      variant='!=')
                                node.children.append(child_rest)
                                queue.append((child_rest, indices_rest))

            if verbose:
                for idx in valid_nodes_indices:
                    node, _ = nodes_to_process[idx]
                    print(f"Depth: {node.depth}, Split: {node.filter_col} {node.filter_val}")

        return root_node

    def predict(self, X: DataFrame) -> np.ndarray:
        check_is_fitted(self)
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.columns_)

        predictions = X.apply(self._predict_row, axis=1)
        return predictions.values

    def _predict_row(self, row: pd.Series) -> float:
        path = []
        curr = self.tree_

        while curr:
            path.append(curr)
            if not curr.children or curr.filter_col is None:
                break

            # Find which child matches
            col = curr.filter_col
            val = curr.filter_val
            row_val = row[col]

            matched_child = None
            if curr.is_numeric:
                if pd.isna(row_val):
                    # Spec doesn't explicitly say what to do if NaN at prediction time for numeric.
                    # "the parcel will still slot into a node slightly higher up the tree"
                    break

                try:
                    # Try to ensure both are numbers.
                    # If val is a string (e.g. "UNKNOWN"), this is actually a categorical split
                    # but is_numeric was set to True. This shouldn't happen with correct training.
                    f_row_val = float(row_val)
                    f_val = float(val)
                    is_less_equal = f_row_val <= f_val
                except (ValueError, TypeError):
                    # Fallback to string comparison or just break if it's completely incompatible
                    try:
                        is_less_equal = row_val <= val
                    except:
                        break  # Cannot compare

                if is_less_equal:
                    matched_child = curr.children[0]
                else:
                    if len(curr.children) > 1:
                        matched_child = curr.children[1]
            else:
                # Categorical
                if pd.isna(row_val):
                    row_val = "NaN"
                else:
                    row_val = str(row_val)

                if row_val == str(val):
                    matched_child = curr.children[0]
                else:
                    if len(curr.children) > 1:
                        matched_child = curr.children[1]

            if matched_child:
                curr = matched_child
            else:
                break

        # Calculate weights
        # path is from root to leaf
        # x = 0 for leaf, x = 1 for root
        # n nodes in path. index i from 0 (root) to n-1 (leaf).
        # x_i = (n - 1 - i) / (n - 1) if n > 1 else 0
        n = len(path)
        if n == 1:
            return path[0].wilson_mean

        total_w = 0
        weighted_sum = 0
        for i, node in enumerate(path):
            x = (n - 1 - i) / (n - 1)
            w = (1 - x) ** self.weight_falloff
            weighted_sum += node.wilson_mean * w
            total_w += w

        return weighted_sum / total_w

    def to_dict(self) -> Dict[str, Any]:
        """
        Exports the trained tree structure as a dictionary.
        """
        check_is_fitted(self)

        def _node_to_dict(node: Optional[CompNode]) -> Optional[Dict[str, Any]]:
            if not node:
                return None

            d = {
                "wilson_mean": float(node.wilson_mean),
                "count": int(node.count),
                "depth": int(node.depth),
                "filter_col": node.filter_col,
                "filter_val": node.filter_val,
                "is_numeric": bool(node.is_numeric),
                "children": [_node_to_dict(child) for child in node.children]
            }
            if node.variant is not None:
                d["variant"] = node.variant

            # Handle non-serializable filter_val (like numpy types)
            if d["filter_val"] is not None:
                if isinstance(d["filter_val"], (np.int64, np.float64, np.int32, np.float32)):
                    d["filter_val"] = d["filter_val"].item()
                elif not isinstance(d["filter_val"], (str, int, float, bool)):
                    d["filter_val"] = str(d["filter_val"])

            return d

        return _node_to_dict(self.tree_)

    def to_json(self, indent: int = 4) -> str:
        """
        Exports the trained tree structure as a JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def explain_value(self, row: Union[DataFrame, pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Audits and traces the path that a row takes through the tree.
        """
        check_is_fitted(self)
        if isinstance(row, pd.Series):
            pass
        elif isinstance(row, dict):
            row = pd.Series(row)
        else:
            # Assume it's a single-row DataFrame or array
            if hasattr(row, 'iloc'):
                row = row.iloc[0]
            else:
                row = pd.Series(row, index=self.columns_)

        path_nodes = []
        curr = self.tree_

        while curr:
            node_info = {
                "depth": curr.depth,
                "wilson_mean": float(curr.wilson_mean),
                "count": int(curr.count),
                "variant": curr.variant,
                "filter_col": curr.filter_col,
                "filter_val": curr.filter_val,
                "is_numeric": curr.is_numeric,
                "actual_value": row[curr.filter_col] if curr.filter_col is not None else None
            }
            path_nodes.append(node_info)

            if not curr.children or curr.filter_col is None:
                break

            col = curr.filter_col
            val = curr.filter_val
            row_val = row[col]

            matched_child = None
            if curr.is_numeric:
                if pd.isna(row_val):
                    break
                try:
                    f_row_val = float(row_val)
                    f_val = float(val)
                    is_less_equal = f_row_val <= f_val
                except (ValueError, TypeError):
                    try:
                        is_less_equal = row_val <= val
                    except:
                        break

                if is_less_equal:
                    matched_child = curr.children[0]
                else:
                    if len(curr.children) > 1:
                        matched_child = curr.children[1]
            else:
                if pd.isna(row_val):
                    row_val = "NaN"
                else:
                    row_val = str(row_val)

                if row_val == str(val):
                    matched_child = curr.children[0]
                else:
                    if len(curr.children) > 1:
                        matched_child = curr.children[1]

            if matched_child:
                curr = matched_child
            else:
                break

        # Calculate weights and final prediction
        n = len(path_nodes)
        if n == 1:
            final_pred = path_nodes[0]["wilson_mean"]
            for node in path_nodes:
                node["weight"] = 1.0
        else:
            total_w = 0
            weighted_sum = 0
            for i, node in enumerate(path_nodes):
                x = (n - 1 - i) / (n - 1)
                w = (1 - x) ** self.weight_falloff
                node["weight"] = float(w)
                weighted_sum += node["wilson_mean"] * w
                total_w += w
            final_pred = weighted_sum / total_w

        # Build calculation string
        calc_parts = [f"({n['wilson_mean']:.2f} * {n['weight']:.4f})" for n in path_nodes]
        calculation_str = f"({' + '.join(calc_parts)}) / {sum(n['weight'] for n in path_nodes):.4f} = {final_pred:.2f}"

        return {
            "final_prediction": float(final_pred),
            "weight_falloff": float(self.weight_falloff),
            "path": path_nodes,
            "calculation": calculation_str
        }
