# Changelog

## [Unreleased]
### Added
- **JSON serialization for fitted models.** `LayeredCompBaggingModel.to_dict`/`from_dict` and
  `to_json`/`from_json`, plus `LayeredCompModel.from_dict` (inverse of the existing `to_dict`). A
  portable, human-readable, pickle-free way to persist a fitted ensemble; round-trips to a
  predict-ready model whose predictions match bit-for-bit (numeric thresholds survive JSON
  losslessly). `from_dict` restores a predict/explain-ready model, not a resume-training one (fit-only
  caches like `pre_sorted_indices_` are not serialized). Format is versioned (`format_version`,
  `lib_version`). Covered by `tests/test_serialization.py`.

### Performance (behavior-neutral — bit-for-bit identical outputs)
- **Vectorized prediction (~20× on large inputs).** `LayeredCompModel.predict` now routes the whole
  row-set down the tree once, partitioning by boolean mask per node and computing the falloff-weighted
  path blend in a single vectorized pass, instead of `DataFrame.apply(_predict_row, axis=1)` row by
  row. Accumulation runs in root→leaf depth order so the float summation reproduces the row-by-row
  result exactly. `_predict_row` is retained as the reference implementation.
- **Tree-level parallelism in bagging fit.** `LayeredCompBaggingModel.fit` draws all per-tree seeds up
  front (preserving the exact RNG sequence) then fits the independent trees across `n_jobs` processes;
  each tree fits with `n_jobs=1` to avoid nested oversubscription. (Gain is modest on Windows due to
  process-spawn/import overhead; larger on fork-based platforms.)
- **Faster split search.** `_find_best_split` replaces a per-column `np.isin(pre_sorted, indices)`
  (O(N) scan) with an O(1) positional boolean membership mask, built once per node.
- New `tests/test_perf_equivalence.py` + `tests/fixtures/baseline_v021.json`: assert the above
  reproduce stock v0.2.1 bit-for-bit (trees, weight_falloffs, predictions; mae & mse), that batched
  predict equals `_predict_row`, and that `n_jobs>1` equals `n_jobs=1`.

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
