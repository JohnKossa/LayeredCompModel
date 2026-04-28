# Changelog

## [0.2.1] - 2026-04-28
### Added
- New bagging quickstart example: `examples/bagging_quickstart.py`.
- New example usage fro LayeredCompBaggingModel in README.md

## [0.2.0] - 2026-04-27
### Added
- `LayeredCompBaggingModel`: A bagging ensemble version of the primary algorithm that reduces variance and automatically optimizes the `weight_falloff` for each tree in the ensemble.
- `src/layeredcompbaggingmodel`: New module for the bagging model.
- Optimization of `weight_falloff`: Using bounded golden method to find the optimal `weight_falloff` (0-15) for each tree based on an internal validation set.
- Reproducibility support: Added `random_state` to `LayeredCompBaggingModel` for consistent ensemble results.

## [0.1.0] - 2026-04-22
### Added
- Initial release: Hierarchical tree-based regressor using path-weighted Wilson means (95% trimmed) for robust predictions (e.g., parcel sale prices).
- NaN handling: Categorical NaNs as distinct "NaN" category (`fillna("NaN").unique()`); numeric NaNs excluded from splits via `notna()` masks (per SPEC.md); target `y` must be finite (raises `ValueError`).
- Scikit-learn compliance: `BaseEstimator`/`RegressorMixin`; works with `Pipeline`, `GridSearchCV`, `cross_val_score`, pickling; partial `check_estimator` pass (intentional NaN trade-off).
- Development: Full type hints (`py.typed`, mypy-ready), 20+ unittest/pytest tests (splits/NaN/explain/pickle/sklearn/bagging), `examples/quickstart.py` (MAE ~127k), `src/` layout, Hatchling build, dev deps (ruff/black/mypy).

Future releases will include Sphinx docs, benchmarks (vs XGBoost/LinearR), CI/CD.
