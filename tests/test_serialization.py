"""JSON serialization round-trip for the fitted ensemble.

A portable, inspectable, pickle-free way to persist a model: fit -> to_dict/to_json -> from_dict/
from_json must yield a predict-ready model whose predictions match the original BIT-FOR-BIT. Covers
numeric splits, categorical one-vs-rest, NaN early-stop, and unseen test categories (via the shared
perf fixture).
"""
import json

import numpy as np
import pytest

import layeredcompmodel
from layeredcompmodel import LayeredCompBaggingModel, LayeredCompModel
from _perf_fixture import make_fixture

METRICS = ("mae", "mse")


@pytest.fixture(scope="module")
def data():
    return make_fixture()


@pytest.mark.parametrize("metric", METRICS)
def test_roundtrip_dict_predicts_identically(data, metric):
    X, X_test, y = data
    bag = LayeredCompBaggingModel(tree_count=6, sample_pct=0.8, random_state=42,
                                  split_metric=metric, n_jobs=1).fit(X, y)
    restored = LayeredCompBaggingModel.from_dict(bag.to_dict())
    assert restored.predict(X_test).tolist() == bag.predict(X_test).tolist()


@pytest.mark.parametrize("metric", METRICS)
def test_roundtrip_json_predicts_identically(data, metric):
    X, X_test, y = data
    bag = LayeredCompBaggingModel(tree_count=5, sample_pct=0.8, random_state=3,
                                  split_metric=metric, n_jobs=1).fit(X, y)
    restored = LayeredCompBaggingModel.from_json(bag.to_json())
    np.testing.assert_array_equal(restored.predict(X_test), bag.predict(X_test))


def test_serialized_dict_is_json_and_carries_metadata(data):
    X, _X_test, y = data
    bag = LayeredCompBaggingModel(tree_count=4, sample_pct=0.8, random_state=1).fit(X, y)
    d = bag.to_dict()
    s = json.dumps(d)                       # must be JSON-serializable (no numpy/objects leaking)
    assert json.loads(s) == d
    assert d["lib_version"] == layeredcompmodel.__version__
    assert d["n_features_in"] == X.shape[1]
    assert len(d["trees"]) == 4
    assert all("tree" in t and "weight_falloff" in t for t in d["trees"])


def test_single_tree_from_dict_predicts(data):
    X, X_test, y = data
    bag = LayeredCompBaggingModel(tree_count=1, sample_pct=0.8, random_state=9).fit(X, y)
    t = bag.estimators_[0]
    state = bag.to_dict()["trees"][0]
    restored = LayeredCompModel.from_dict(state)
    np.testing.assert_array_equal(restored.predict(X_test), t.predict(X_test))


def test_from_dict_rejects_unknown_format(data):
    X, _X_test, y = data
    bag = LayeredCompBaggingModel(tree_count=2, random_state=1).fit(X, y)
    bad = bag.to_dict()
    bad["format_version"] = 999
    with pytest.raises(ValueError, match="format_version"):
        LayeredCompBaggingModel.from_dict(bad)
