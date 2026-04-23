# LayeredCompModel

[![PyPI version](https://badge.fury.io/py/layeredcompmodel.svg)](https://pypi.org/project/layeredcompmodel/)
[![Documentation Status](https://readthedocs.org/projects/layeredcompmodel/badge/?version=latest)](https://layeredcompmodel.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Hierarchical tree-based regressor for robust predictions (e.g., parcel sale prices) using path-weighted Wilson means (95% trimmed means for outlier resistance).

* [MODEL_SPEC.md](MODEL_SPEC.md): High-level method.
* [SPEC.md](SPEC.md): Detailed implementation specs.

## Features

- **Scikit-learn compatible**: Inherits `BaseEstimator`/`RegressorMixin`; works with `Pipeline`, `GridSearchCV`, `cross_val_score`, pickling.
- **Automatic feature handling**: Categorical (one-vs-rest splits), numeric (binary search breakpoints), NaNs/missing values.
- **Robust statistics**: Wilson means prevent outlier swings.
- **Configurable weighting**: `weight_falloff` balances local accuracy vs. market normativity.
- **Explainable**: `explain_value(row)` shows path, weights, means.
- **Serializable**: `to_json()`, `to_dict()`.
- **Parallel**: `n_jobs` support.

### NaN Handling
- **Categorical**: Treated as distinct "NaN" category.
- **Numeric**: Excluded from splits (robust; per SPEC.md).
- **Target `y`**: Must be finite (raises `ValueError`).
- Strict checks: Use `Pipeline([('imputer', SimpleImputer()), ('model', LayeredCompModel())])`.

## Installation

```bash
pip install layeredcompmodel
```

For development:

```bash
git clone https://github.com/JohnKossa/layeredcompmodel.git
cd layeredcompmodel
pip install -e .[dev]
```

## Quickstart

```python
import pandas as pd
import numpy as np
from layeredcompmodel import LayeredCompModel

# Synthetic real-estate-like data
rng = np.random.default_rng(42)
n_samples = 100
data = {
    'neighborhood': rng.choice(['North', 'South', 'East'], n_samples),
    'size_sqft': rng.normal(2000, 500, n_samples),
    'price': rng.normal(500000, 100000, n_samples) + 100 * rng.normal(0, 1, n_samples) * (rng.normal(0, 1, n_samples) * 2000)
}
df = pd.DataFrame(data)
X = df[['neighborhood', 'size_sqft']]
y = df['price']

# Train
model = LayeredCompModel(weight_falloff=0.8, n_jobs=1)
model.fit(X, y)

# Predict
predictions = model.predict(X)
print(f&quot;Predictions shape: {predictions.shape}&quot;)
print(f&quot;MAE: {np.mean(np.abs(predictions - y)):.0f}&quot;)

# Explain single prediction
explanation = model.explain_value(X.iloc[0:1].squeeze())
print(explanation)
```

## API Reference

### LayeredCompModel(weight_falloff=0.5, split_metric='mae', n_jobs=1)

- `fit(X, y)`: Build tree from features `X` (DataFrame), target `y` (Series).
- `predict(X)`: Predict using path-weighted means.
- `explain_value(row)`: Dict with path nodes, depths, weights, wilson_means.
- `to_json(indent=4)`: JSON tree dump.
- `tree_`: Root `CompNode` (filter_col, filter_val, wilson_mean, children).

See [docs](https://layeredcompmodel.readthedocs.io) (TBD).

## Examples

See [`examples/quickstart.py`](examples/quickstart.py) for a runnable example (code matches Quickstart above).

**Run it:**
```bash
python examples/quickstart.py
```

**Expected output:**
```
Predictions shape: (100,)
MAE: 126914
{'final_prediction': 530354.0426294187, 'weight_falloff': 0.8, 'path': [{'depth': 0, 'wilson_mean': 476353.91361128056, 'count': 100, 'is_leaf': False, 'filter_col': 'size_sqft', 'filter_val': 2101.366485546922}, {'depth': 1, 'wilson_mean': 553953.0606894617, 'count': 42, 'is_leaf': False, 'filter_col': 'neighborhood', 'filter_val': 'North'}, {'depth': 2, 'wilson_mean': 525096.3185716979, 'count': 13, 'is_leaf': True}], 'calculation': '0.199*476354 + 0.512*553953 + 0.289*525096 = 530354'}
```

## Development & Testing

```bash
pytest tests/ --cov=layeredcompmodel
black src/
mypy src/
```

CI/CD, Sphinx docs: planned.

## Citing

Kossa, J. (2026). LayeredCompModel. GitHub. https://github.com/JohnKossa/layeredcompmodel

## License

[MIT](LICENSE)