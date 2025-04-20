import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import os

# Set up paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
predictions_path = os.path.join(PROJECT_ROOT, 'models', 'ml_predictions.csv')

# Read predictions
predictions = pd.read_csv(predictions_path)
predictions['Date'] = pd.to_datetime(predictions['Date'])
predictions.set_index('Date', inplace=True)

# Calculate residuals
predictions['residuals'] = predictions['y_true'] - predictions['y_pred']

# Calculate naive forecast (previous value)
predictions['naive_pred'] = predictions['y_true'].shift(1)

# Create figure with subplots
fig = plt.figure(figsize=(20, 15))

# 1. Residuals Time Plot
ax1 = plt.subplot(311)
ax1.plot(predictions.index, predictions['residuals'], 'b-', label='Residuals')
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.3)
ax1.fill_between(predictions.index, 
                 predictions['residuals'], 
                 0, 
                 alpha=0.3, 
                 color='blue')
ax1.set_title('Residuals Over Time', fontsize=12)
ax1.set_ylabel('Residual (y_true - y_pred)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. ACF/PACF Plot
ax2 = plt.subplot(312)
lags = min(20, len(predictions) - 1)
acf_values = acf(predictions['residuals'].dropna(), nlags=lags)
lags_x = np.arange(len(acf_values))
ax2.bar(lags_x, acf_values, alpha=0.5)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
ax2.axhline(y=1.96/np.sqrt(len(predictions)), color='g', linestyle=':', alpha=0.3)
ax2.axhline(y=-1.96/np.sqrt(len(predictions)), color='g', linestyle=':', alpha=0.3)
ax2.set_title('Autocorrelation Function (ACF) of Residuals', fontsize=12)
ax2.set_xlabel('Lag')
ax2.grid(True, alpha=0.3)

# 3. Prediction Comparison
ax3 = plt.subplot(313)
ax3.plot(predictions.index, predictions['y_true'], 'b-', label='Actual (log)', linewidth=2)
ax3.plot(predictions.index, predictions['y_pred'], 'r--', label='Ridge Prediction', linewidth=2)
ax3.plot(predictions.index, predictions['naive_pred'], 'g:', label='Naive Forecast', linewidth=2)
ax3.set_title('Model Predictions vs Actual Values', fontsize=12)
ax3.set_ylabel('Log Price')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()

# Save plots
output_dir = os.path.join(PROJECT_ROOT, 'reports', 'figures')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'residual_analysis.png'), 
            dpi=300, 
            bbox_inches='tight')

# Calculate additional statistics
print("\nResidual Analysis Statistics:")
print("-" * 30)
print(f"Mean Residual: {predictions['residuals'].mean():.4f}")
print(f"Std Residual: {predictions['residuals'].std():.4f}")
print(f"Skewness: {predictions['residuals'].skew():.4f}")
print(f"Kurtosis: {predictions['residuals'].kurtosis():.4f}")

# Calculate autocorrelation at different lags
print("\nAutocorrelation at Different Lags:")
print("-" * 30)
for lag in [1, 2, 3]:
    acf_value = acf(predictions['residuals'].dropna(), nlags=lag)[-1]
    print(f"Lag {lag}: {acf_value:.4f}")

# Calculate error statistics vs naive forecast
ridge_mse = ((predictions['y_true'] - predictions['y_pred'])**2).mean()
naive_mse = ((predictions['y_true'] - predictions['naive_pred'])**2).mean()

print("\nComparison with Naive Forecast:")
print("-" * 30)
print(f"Ridge MSE: {ridge_mse:.4f}")
print(f"Naive MSE: {naive_mse:.4f}")
print(f"Relative Performance: {(naive_mse/ridge_mse - 1)*100:.2f}% improvement over naive")