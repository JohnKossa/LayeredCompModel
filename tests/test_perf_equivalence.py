"""Equivalence guarantees for the perf work (vectorized predict, tree-level parallelism, faster
split search). These lock in the promise: **pure speed, zero behavior change.**

`tests/fixtures/baseline_v021.json` was captured from stock v0.2.1 (predictions + falloffs + full
tree structures, for mae and mse) on the deterministic fixture in `_perf_fixture.py`. The perf
changes must reproduce it bit-for-bit.
"""
import json
import os

import numpy as np
import pytest

from layeredcompmodel import LayeredCompBaggingModel, LayeredCompModel
from _perf_fixture import make_fixture

BASELINE = os.path.join(os.path.dirname(__file__), "fixtures", "baseline_v021.json")
METRICS = ("mae", "mse")


@pytest.fixture(scope="module")
def data():
    return make_fixture()


@pytest.fixture(scope="module")
def baseline():
    with open(BASELINE) as f:
        return json.load(f)


@pytest.mark.parametrize("metric", METRICS)
def test_fit_matches_stock_baseline_bit_for_bit(data, baseline, metric):
    """Trees, per-tree weight_falloffs, and predictions all reproduce stock v0.2.1 exactly."""
    X, X_test, y = data
    bag = LayeredCompBaggingModel(tree_count=6, sample_pct=0.8, random_state=42,
                                  split_metric=metric, n_jobs=1)
    bag.fit(X, y)
    b = baseline[metric]
    assert [t.to_dict() for t in bag.estimators_] == b["trees"]            # identical tree structure
    assert [t.weight_falloff for t in bag.estimators_] == b["falloffs"]    # identical learned falloffs
    assert bag.predict(X_test).tolist() == b["preds"]                      # identical predictions


@pytest.mark.parametrize("metric", METRICS)
def test_batched_predict_equals_rowwise(data, metric):
    """The vectorized predict equals the reference row-by-row `_predict_row` on inputs that exercise
    numeric splits, categorical one-vs-rest, NaN early-stop, and unseen-category missing-child."""
    X, X_test, y = data
    bag = LayeredCompBaggingModel(tree_count=6, sample_pct=0.8, random_state=42,
                                  split_metric=metric, n_jobs=1)
    bag.fit(X, y)
    for tree in bag.estimators_:
        batched = tree.predict(X_test)
        rowwise = X_test.apply(tree._predict_row, axis=1).to_numpy()
        np.testing.assert_allclose(batched, rowwise, rtol=0, atol=1e-9)


@pytest.mark.parametrize("metric", METRICS)
def test_parallel_matches_serial(data, metric):
    """Tree-level parallelism (n_jobs>1) yields the identical ensemble as serial (n_jobs=1)."""
    X, _X_test, y = data
    serial = LayeredCompBaggingModel(tree_count=6, sample_pct=0.8, random_state=42,
                                     split_metric=metric, n_jobs=1).fit(X, y)
    parallel = LayeredCompBaggingModel(tree_count=6, sample_pct=0.8, random_state=42,
                                       split_metric=metric, n_jobs=4).fit(X, y)
    assert [t.to_dict() for t in serial.estimators_] == [t.to_dict() for t in parallel.estimators_]
    assert [t.weight_falloff for t in serial.estimators_] == [t.weight_falloff for t in parallel.estimators_]


def test_determinism(data):
    """Same random_state => identical model across independent fits."""
    X, X_test, y = data
    a = LayeredCompBaggingModel(tree_count=5, sample_pct=0.8, random_state=7).fit(X, y)
    b = LayeredCompBaggingModel(tree_count=5, sample_pct=0.8, random_state=7).fit(X, y)
    assert a.predict(X_test).tolist() == b.predict(X_test).tolist()


def test_predict_empty(data):
    X, _X_test, y = data
    bag = LayeredCompBaggingModel(tree_count=3, sample_pct=0.8, random_state=1).fit(X, y)
    assert bag.predict(X.iloc[0:0]).shape == (0,)
