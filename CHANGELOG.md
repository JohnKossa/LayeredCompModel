# Changelog

## [0.1.0] - 2026-04-22
### Added
- Initial release: Hierarchical tree-based regressor using path-weighted Wilson means (95% trimmed) for robust predictions (e.g., parcel sale prices).
- NaN handling: Categorical NaNs as distinct "NaN" category (`fillna("NaN").unique()`); numeric NaNs excluded from splits via `notna()` masks (per SPEC.md); target `y` must be finite (raises `ValueError`).
- Scikit-learn compliance: `BaseEstimator`/`RegressorMixin`; works with `Pipeline`, `GridSearchCV`, `cross_val_score`, pickling; partial `check_estimator` pass (intentional NaN trade-off).
- Development: Full type hints (`py.typed`, mypy-ready), 16+ unittest/pytest tests (splits/NaN/explain/pickle/sklearn), `examples/quickstart.py` (MAE ~127k), `src/` layout, Hatchling build, dev deps (ruff/black/mypy).

Future releases will include Sphinx docs, benchmarks (vs XGBoost/LinearR), CI/CD.
