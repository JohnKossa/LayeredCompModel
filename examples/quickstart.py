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
print(f"Predictions shape: {predictions.shape}")
print(f"MAE: {np.mean(np.abs(predictions - y)):.0f}")

# Explain single prediction
explanation = model.explain_value(X.iloc[0:1].squeeze())
print(explanation)