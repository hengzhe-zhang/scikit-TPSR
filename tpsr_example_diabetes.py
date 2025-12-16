"""Simple example using TPSRRegressor on sklearn diabetes dataset."""

import sys
from pathlib import Path

# Add nesymres/src to Python path for imports
nesymres_src = Path(__file__).parent / "nesymres" / "src"
if str(nesymres_src) not in sys.path:
    sys.path.insert(0, str(nesymres_src))

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tpsr_sklearn_wrapper import TPSRRegressor

# Load diabetes dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and fit TPSR model
model = TPSRRegressor(device="cpu", max_input_points=200, width=3, horizon=200)
print("Fitting TPSR model...")
model.fit(X_train, y_train, verbose=True)

# Make predictions
print("\nMaking predictions...")
y_pred = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\nResults:")
print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
