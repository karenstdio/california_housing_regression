# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

# Display plots inline
import warnings
warnings.filterwarnings('ignore')

# Load California Housing dataset (20k+ samples, 8 numeric features)
data = fetch_california_housing(as_frame=True)

X = data.data                    # Features (8 numeric columns)
y = data.target * 100000         # Target (house price) – scaled to $100k

print(f"Data shape: {X.shape}")
X.head()

# Split data: 80% training, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale numeric features with StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Predictions
y_pred_lin = lin_reg.predict(X_test_scaled)

# Evaluation metrics
mse_lin  = mean_squared_error(y_test, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)
r2_lin   = r2_score(y_test, y_pred_lin)

print(f"Linear Regression – MSE: {mse_lin:.0f}, RMSE: {rmse_lin:.0f}, R²: {r2_lin:.3f}")

# Define alpha values between 0.1 and 10 (log-scale)
alphas = np.logspace(-1, 1, 20)

# Ridge Regression with cross-validation
ridge = RidgeCV(alphas=alphas, cv=5)
ridge.fit(X_train_scaled, y_train)

print("Best selected alpha:", ridge.alpha_)

# Ridge predictions
y_pred_ridge = ridge.predict(X_test_scaled)

mse_ridge  = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
r2_ridge   = r2_score(y_test, y_pred_ridge)

print(f"Ridge Regression – MSE: {mse_ridge:.0f}, RMSE: {rmse_ridge:.0f}, R²: {r2_ridge:.3f}")

# Scatter plot: True vs Predicted values
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_lin, alpha=0.4, label='Linear')
plt.scatter(y_test, y_pred_ridge, alpha=0.4, label='Ridge')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('True Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('True vs Predicted Prices')
plt.legend()
plt.tight_layout()
plt.show()

# Residual distribution
residuals = y_test - y_pred_ridge

plt.figure(figsize=(8,4))
sns.histplot(residuals, bins=40, kde=True)
plt.title('Ridge Regression Residual Distribution')
plt.xlabel('Residuals ($)')
plt.show()
