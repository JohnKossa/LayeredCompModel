import pandas as pd
import numpy as np
from layeredcompmodel import LayeredCompBaggingModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# Synthetic real-estate-like data
rng = np.random.default_rng(42)
n_samples = 200
data = {
    'neighborhood': rng.choice(['North', 'South', 'East', 'West'], n_samples),
    'size_sqft': rng.normal(2000, 500, n_samples),
    'year_built': rng.integers(1950, 2023, n_samples),
    'price': rng.normal(500000, 100000, n_samples)
}
df = pd.DataFrame(data)

# Add some logic to price based on features
df['price'] += (df['neighborhood'] == 'North') * 60000
df['price'] += df['size_sqft'] * 325
df['price'] += (df['year_built'] - 1950) * 1200
df['price'] *= (rng.normal(1, 0.3))

X = df[['neighborhood', 'size_sqft', 'year_built']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Bagging Model
# It will build 5 trees, each on 80% of the data
# And find the best weight_falloff for each tree automatically
print("Training LayeredCompBaggingModel...")
model = LayeredCompBaggingModel(tree_count=10, sample_pct=0.95, random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mape = 100 * mean_absolute_percentage_error(y_test, predictions)

print(f"\nResults for {len(y_test)} test samples:")
print(f"MAE: {mae:.0f}")
print(f"MAPE: {mape: .2f}%")

# Show the optimized weight_falloff values
print("\nOptimized weight_falloffs per tree:")
for i, tree in enumerate(model.estimators_):
    print(f"Tree {i+1}: {tree.weight_falloff:.4f}")
