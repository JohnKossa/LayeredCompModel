import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from layeredcompmodel import LayeredCompBaggingModel

def test_bagging_model_basic():
    X, y = make_regression(n_samples=50, n_features=4, random_state=42)
    X = pd.DataFrame(X, columns=['a', 'b', 'c', 'd'])
    y = pd.Series(y)
    
    model = LayeredCompBaggingModel(tree_count=5, sample_pct=0.8, random_state=42)
    # print(X)
    # print(y)
    model.fit(X, y)
    
    assert len(model.estimators_) == 5
    
    preds = model.predict(X)
    assert len(preds) == len(y)
    assert np.all(np.isfinite(preds))

def test_bagging_model_hyperparameters():
    X, y = make_regression(n_samples=20, n_features=2, random_state=42)
    
    # Test invalid tree_count
    with pytest.raises(ValueError, match="tree_count must be >= 1"):
        LayeredCompBaggingModel(tree_count=0).fit(X, y)
        
    # Test invalid sample_pct
    with pytest.raises(ValueError, match="sample_pct must be between 0 and 1"):
        LayeredCompBaggingModel(sample_pct=1.0).fit(X, y)
    
    with pytest.raises(ValueError, match="sample_pct must be between 0 and 1"):
        LayeredCompBaggingModel(sample_pct=0.0).fit(X, y)

def test_bagging_model_split_metric():
    X, y = make_regression(n_samples=20, n_features=2, random_state=42)
    
    model_mse = LayeredCompBaggingModel(tree_count=2, split_metric='mse', random_state=42)
    model_mse.fit(X, y)
    assert model_mse.split_metric == 'mse'
    
    model_mae = LayeredCompBaggingModel(tree_count=2, split_metric='mae', random_state=42)
    model_mae.fit(X, y)
    assert model_mae.split_metric == 'mae'

def test_bagging_model_small_data():
    # Test with very small dataset
    X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    y = pd.Series([10, 20, 30])
    
    model = LayeredCompBaggingModel(tree_count=2, sample_pct=0.5, random_state=42)
    model.fit(X, y)
    
    preds = model.predict(X)
    assert len(preds) == 3

def test_bagging_model_random_state():
    X, y = make_regression(n_samples=50, n_features=4, random_state=42)
    
    model1 = LayeredCompBaggingModel(tree_count=3, random_state=42)
    model1.fit(X, y)
    
    model2 = LayeredCompBaggingModel(tree_count=3, random_state=42)
    model2.fit(X, y)
    
    np.testing.assert_array_almost_equal(model1.predict(X), model2.predict(X))
    
    model3 = LayeredCompBaggingModel(tree_count=3, random_state=43)
    model3.fit(X, y)
    
    with pytest.raises(AssertionError):
        np.testing.assert_array_almost_equal(model1.predict(X), model3.predict(X))
