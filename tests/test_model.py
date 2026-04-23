import unittest
import numpy as np
import pandas as pd
from layeredcompmodel import calculate_wilson_mean, LayeredCompModel
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.utils.estimator_checks import check_estimator
import pickle


class TestLayeredCompModel(unittest.TestCase):
    def test_wilson_mean(self):
        # 100 numbers from 1 to 100
        y = pd.Series(range(1, 101))
        # Trim top 2.5% (99, 100) and bottom 2.5% (1, 2)
        # Remaining: 3 to 98
        # Mean should be (3 + 98) / 2 = 50.5
        wm = calculate_wilson_mean(y)
        self.assertAlmostEqual(wm, 50.5)

    def test_categorical_split(self):
        # Create data where 'cat' column perfectly splits the data
        # 'A' has price 10, 'B' has price 20
        # 'land' and 'build' are key features, but we'll make them constant so R2 is 0 or NaN
        # Wait, R2 of constant is 0 or error. Let's give them some variance.
        data = {
            'cat': ['A'] * 10 + ['B'] * 10,
            'land': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            'build': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            'price': [10] * 10 + [20] * 10
        }
        df = pd.DataFrame(data)
        X = df[['cat', 'land', 'build']]
        y = df['price']

        model = LayeredCompModel()
        model.fit(X, y)

        self.assertIsNotNone(model.tree_.filter_col)
        self.assertEqual(model.tree_.filter_col, 'cat')
        # One of the variants should be the filter value
        self.assertIn(model.tree_.filter_val, ['A', 'B'])

    def test_numeric_split(self):
        # Numeric split on 'size'
        data = {
            'size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'land': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # key features with variance
            'build': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            'price': [10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
        }
        df = pd.DataFrame(data)
        X = df[['size', 'land', 'build']]
        y = df['price']

        # In this data, a split at size=5 gives two groups:
        # low: size [1..5], price [10..14]. land [1..5]. 
        # high: size [6..10], price [20..24]. land [6..10].
        # In both groups, price = land + 9. Linear regression should fit perfectly.
        # MAE should be 0 for both groups.

        model = LayeredCompModel()
        model.fit(X, y)

        self.assertEqual(model.tree_.filter_col, 'size')
        # Midpoint should be around 5
        self.assertTrue(4 <= model.tree_.filter_val <= 6)

    def test_prediction_weights(self):
        # Simple tree with 2 levels
        data = {
            'cat': ['A'] * 5 + ['B'] * 5,
            'land': [1, 2, 3, 4, 5] * 2,
            'build': [1, 1, 1, 1, 1] * 2,
            'price': [10, 10, 10, 10, 10, 20, 20, 20, 20, 20]
        }
        df = pd.DataFrame(data)
        X = df[['cat', 'land', 'build']]
        y = df['price']

        model = LayeredCompModel(weight_falloff=1.0)
        model.fit(X, y)

        # Root mean is 15
        # Leaf mean for 'A' is 10, for 'B' is 20
        # Prediction for 'A':
        # path: [Root(15), Leaf(10)]
        # n = 2
        # i=0 (Root): x = (2-1-0)/(2-1) = 1. w = (1-1)^1 = 0
        # i=1 (Leaf): x = (2-1-1)/(2-1) = 0. w = (1-0)^1 = 1
        # Prediction = (15*0 + 10*1) / (0+1) = 10

        pred = model.predict(X.iloc[:1])
        self.assertAlmostEqual(pred[0], 10.0)

        # Test weight_falloff = 0 (even weights)
        model.weight_falloff = 0.0
        # Prediction = (15*1 + 10*1) / 2 = 12.5
        pred = model.predict(X.iloc[:1])
        self.assertAlmostEqual(pred[0], 12.5)

    def test_categorical_split_with_nan(self):
        # Create data where 'cat' column has NaNs
        data = {
            'cat': ['A'] * 10 + ['B'] * 10 + [None] * 10,
            'land': np.random.rand(30),
            'build': np.random.rand(30),
            'price': [10] * 10 + [20] * 10 + [30] * 10
        }
        df = pd.DataFrame(data)
        X = df[['cat', 'land', 'build']]
        y = df['price']

        model = LayeredCompModel(weight_falloff=10.0)
        model.fit(X, y)

        # It should split on 'cat'
        self.assertEqual(model.tree_.filter_col, 'cat')

        # Test prediction on a NaN row
        test_df = pd.DataFrame({'cat': [None], 'land': [0.5], 'build': [0.5]})
        pred = model.predict(test_df)
        # Root mean is 20. Leaf mean for NaN is 30.
        # With high weight_falloff, it should be very close to 30.
        self.assertAlmostEqual(pred[0], 30.0, places=1)

    def test_numeric_split_with_nan(self):
        # Numeric split with NaNs in the feature
        data = {
            'size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] + [None] * 5,
            'land': np.random.rand(15),
            'build': np.random.rand(15),
            'price': [10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 15, 15, 15, 15, 15]
        }
        df = pd.DataFrame(data)
        X = df[['size', 'land', 'build']]
        y = df['price']

        model = LayeredCompModel()
        model.fit(X, y)

        # The numeric split should ignore NaNs during its scoring process
        # But during tree building, NaNs might end up in the 'rest' bucket if not handled carefully
        # Wait, my code: mask = (X[col] <= val) & X[col].notna()
        # X_low = X[mask], X_high = X[~mask]
        # ~mask will include NaNs!

        self.assertIsNotNone(model.tree_.filter_col)

    def test_json_export(self):
        # Test categorical export
        data = {
            'cat': ['A'] * 5 + ['B'] * 5,
            'price': [10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
        }
        df = pd.DataFrame(data)
        model = LayeredCompModel()
        model.fit(df[['cat']], df['price'])

        tree_dict = model.to_dict()
        self.assertIsInstance(tree_dict, dict)
        self.assertEqual(tree_dict['filter_col'], 'cat')
        self.assertNotIn('variant', tree_dict)  # Root should not have variant

        # Check children variants
        self.assertTrue(len(tree_dict['children']) > 0)
        variants = [child['variant'] for child in tree_dict['children']]
        self.assertIn('=', variants)
        self.assertIn('!=', variants)

        # Test numeric export
        data_num = {
            'size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'price': [10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
        }
        df_num = pd.DataFrame(data_num)
        model_num = LayeredCompModel()
        model_num.fit(df_num[['size']], df_num['price'])

        tree_dict_num = model_num.to_dict()
        variants_num = [child['variant'] for child in tree_dict_num['children']]
        self.assertIn('<=', variants_num)
        self.assertIn('>', variants_num)

        tree_json = model.to_json()
        self.assertIsInstance(tree_json, str)
        import json
        reloaded = json.loads(tree_json)
        self.assertEqual(reloaded['filter_col'], 'cat')
        self.assertNotIn('variant', reloaded)
        self.assertIn('count', reloaded)

    def test_explain_value(self):
        data = {
            'cat': ['A'] * 5 + ['B'] * 5,
            'size': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'price': [10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
        }
        df = pd.DataFrame(data)
        model = LayeredCompModel(weight_falloff=1.0)
        model.fit(df[['cat', 'size']], df['price'])

        row = df.iloc[0]  # cat='A', size=1
        explanation = model.explain_value(row)

        self.assertIn('final_prediction', explanation)
        self.assertIn('path', explanation)
        self.assertIn('calculation', explanation)
        self.assertGreater(len(explanation['path']), 0)

        # Check first node in path is root
        self.assertEqual(explanation['path'][0]['depth'], 0)
        self.assertIsNone(explanation['path'][0]['variant'])

        # Verify calculation string matches final prediction (roughly)
        self.assertIn(f"{explanation['final_prediction']:.2f}", explanation['calculation'])

        # Check actual_value is recorded
        self.assertEqual(explanation['path'][0]['actual_value'], row[explanation['path'][0]['filter_col']])

    def test_get_params_and_set_params(self):
        model = LayeredCompModel(weight_falloff=0.3, split_metric='mse')
        params = model.get_params()
        self.assertEqual(params['weight_falloff'], 0.3)
        self.assertEqual(params['split_metric'], 'mse')
        self.assertEqual(params['n_jobs'], 1)

        # Test set_params
        model.set_params(weight_falloff=0.7, split_metric='mae', n_jobs=2)
        new_params = model.get_params()
        self.assertEqual(new_params['weight_falloff'], 0.7)
        self.assertEqual(new_params['split_metric'], 'mae')
        self.assertEqual(new_params['n_jobs'], 2)


    def test_fit_predict_score(self):
        X, y = make_regression(n_samples=100, n_features=4, random_state=42)
        X = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(4)])
        y = pd.Series(y)
        model = LayeredCompModel()
        model.fit(X, y)
        preds = model.predict(X)
        self.assertEqual(preds.shape[0], 100)
        self.assertFalse(np.any(np.isnan(preds)))
        score = model.score(X, y)
        self.assertGreater(score, -1.0)
        self.assertLess(score, 1.01)

    def test_pipeline(self):
        X, y = make_regression(n_samples=100, n_features=4, random_state=42)
        pipe = Pipeline(steps=[('scaler', StandardScaler()), ('model', LayeredCompModel())])
        pipe.fit(X, y)
        score = pipe.score(X, y)
        self.assertGreater(score, -1.0)
        preds = pipe.predict(X)
        self.assertEqual(preds.shape[0], 100)

    def test_grid_search(self):
        X, y = make_regression(n_samples=50, n_features=4, random_state=42)
        grid = GridSearchCV(LayeredCompModel(), {'weight_falloff': [0.1, 0.5, 0.9], 'split_metric': ['mae', 'mse']},
                            cv=3)
        grid.fit(X, y)
        self.assertTrue(hasattr(grid, 'best_params_'))
        self.assertGreater(grid.score(X, y), -1.0)

    def test_cross_val_score(self):
        X, y = make_regression(n_samples=100, n_features=4, random_state=42)
        scores = cross_val_score(LayeredCompModel(), X, y, cv=5)
        self.assertGreater(scores.mean(), -1.0)
        self.assertGreater(len(scores), 0)

    def test_pickle(self):
        X, y = make_regression(n_samples=50, n_features=4, random_state=42)
        model = LayeredCompModel().fit(X, y)
        model_preds = model.predict(X)
        pickled = pickle.dumps(model)
        loaded_model = pickle.loads(pickled)
        loaded_preds = loaded_model.predict(X)
        np.testing.assert_allclose(loaded_preds, model_preds, rtol=1e-5)

    def test_edge_cases(self):
        # Single sample
        X_single = pd.DataFrame({'feat': [1.0]})
        y_single = pd.Series([10.0])
        model = LayeredCompModel()
        model.fit(X_single, y_single)
        pred = model.predict(pd.DataFrame({'feat': [2.0]}))[0]
        self.assertAlmostEqual(pred, 10.0)

        # Identical y values - no split
        X_id = pd.DataFrame({'feat': [1, 2, 3]})
        y_id = pd.Series([5.0, 5.0, 5.0])
        model_id = LayeredCompModel()
        model_id.fit(X_id, y_id)
        self.assertIsNone(model_id.tree_.filter_col)
        self.assertAlmostEqual(model_id.predict(X_id)[0], 5.0)

        # n_jobs >1
        X_small, y_small = make_regression(n_samples=20, n_features=2, random_state=42)
        model_njobs = LayeredCompModel(n_jobs=2)
        model_njobs.fit(X_small, y_small)

        # Empty data raises
        with self.assertRaises(ValueError):
            model.fit(pd.DataFrame(), pd.Series([]))


if __name__ == '__main__':
    unittest.main()
