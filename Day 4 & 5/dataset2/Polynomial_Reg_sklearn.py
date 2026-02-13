"""Polynomial_Reg_sklearn.py

Example using scikit-learn `Pipeline` with `PolynomialFeatures` and `Ridge`.

This script loads training and test CSV files, fits polynomial ridge models
over several regularization values, compares performance, and saves plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ==============================
# Load Datasets
# ==============================
train25 = pd.read_csv(r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset2\Train-2a-25.csv")
train100 = pd.read_csv(r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset2\Train-2b-100.csv")
test = pd.read_csv(r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset2\Test-50.csv")

# Extract features and targets
X25 = train25[["x1", "x2"]].values
y25 = train25["output"].values

X100 = train100[["x1", "x2"]].values
y100 = train100["output"].values

X_test = test[["x1", "x2"]].values
y_test = test["output"].values

print("Data loaded successfully!")
print(f"Train25 shape: {X25.shape}, Train100 shape: {X100.shape}, Test shape: {X_test.shape}")

# ==============================
# Create Pipeline for Polynomial Regression
# ==============================
def create_poly_pipeline(degree=2, alpha=1.0):
    """
    Creates a sklearn Pipeline with PolynomialFeatures and Ridge regression
    
    Args:
        degree: Polynomial degree (default=2)
        alpha: Ridge regularization parameter (default=1.0)
    
    Returns:
        Pipeline object
    """
    pipeline = Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree, include_bias=True)),
        ('scaler', StandardScaler()),
        ('ridge_regression', Ridge(alpha=alpha))
    ])
    return pipeline

# ==============================
# Train models with different datasets and alpha values
# ==============================
degree = 2
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

print("\n" + "="*70)
print("Training Polynomial Regression Models (Degree 2)")
print("="*70)

best_model = None
best_rmse = float('inf')
best_alpha = None
best_dataset = None

results = {
    'train25': {},
    'train100': {}
}

# Train on 25 samples
print("\n--- Training on 25 samples ---")
for alpha in alphas:
    model = create_poly_pipeline(degree=degree, alpha=alpha)
    model.fit(X25, y25)
    
    pred_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    r2 = r2_score(y_test, pred_test)
    
    results['train25'][alpha] = {'model': model, 'rmse': rmse, 'r2': r2, 'pred': pred_test}
    
    print(f"Alpha = {alpha:7.4f} | RMSE = {rmse:10.4f} | R² = {r2:7.4f}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha
        best_model = model
        best_dataset = 'train25'

# Train on 100 samples
print("\n--- Training on 100 samples ---")
for alpha in alphas:
    model = create_poly_pipeline(degree=degree, alpha=alpha)
    model.fit(X100, y100)
    
    pred_test = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))
    r2 = r2_score(y_test, pred_test)
    
    results['train100'][alpha] = {'model': model, 'rmse': rmse, 'r2': r2, 'pred': pred_test}
    
    print(f"Alpha = {alpha:7.4f} | RMSE = {rmse:10.4f} | R² = {r2:7.4f}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha
        best_model = model
        best_dataset = 'train100'

# ==============================
# Display Best Model Results
# ==============================
print("\n" + "="*70)
print(f"BEST MODEL: Trained on {best_dataset} | Alpha = {best_alpha} | RMSE = {best_rmse:.4f}")
print("="*70)

best_pred = best_model.predict(X_test)

print("\nTop 15 Predictions vs Actual Values:")
print(f"{'Index':<8}{'Predicted':<15}{'Actual':<15}{'Error':<15}{'Abs Error':<15}")
print("-" * 68)
for i in range(min(15, len(y_test))):
    error = best_pred[i] - y_test[i]
    abs_error = abs(error)
    print(f"{i:<8}{best_pred[i]:<15.4f}{y_test[i]:<15.4f}{error:<15.4f}{abs_error:<15.4f}")

# ==============================
# Visualization
# ==============================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Polynomial Regression: Dataset Comparison', fontsize=16, fontweight='bold')

# Plot 1: Best model predictions vs actual (all test samples)
ax1 = axes[0, 0]
indices = np.arange(len(y_test))
ax1.scatter(indices, y_test, label='Actual', color='blue', s=50, alpha=0.7, edgecolors='black')
ax1.scatter(indices, best_pred, label='Predicted', color='red', s=50, alpha=0.7, edgecolors='black', marker='^')
ax1.plot(indices, y_test, 'b--', alpha=0.3)
ax1.plot(indices, best_pred, 'r--', alpha=0.3)
ax1.set_xlabel('Test Sample Index', fontsize=11)
ax1.set_ylabel('Output Value', fontsize=11)
ax1.set_title(f'Best Model ({best_dataset}, α={best_alpha})', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Predicted vs Actual scatter plot
ax2 = axes[0, 1]
ax2.scatter(y_test, best_pred, color='green', s=60, alpha=0.6, edgecolors='black')
min_val = min(y_test.min(), best_pred.min())
max_val = max(y_test.max(), best_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
ax2.set_xlabel('Actual Values', fontsize=11)
ax2.set_ylabel('Predicted Values', fontsize=11)
ax2.set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals (errors)
ax3 = axes[1, 0]
residuals = y_test - best_pred
ax3.scatter(best_pred, residuals, color='purple', s=60, alpha=0.6, edgecolors='black')
ax3.axhline(y=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Predicted Values', fontsize=11)
ax3.set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
ax3.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: RMSE comparison across alpha values
ax4 = axes[1, 1]
alphas_list = list(results['train25'].keys())
rmse_25 = [results['train25'][a]['rmse'] for a in alphas_list]
rmse_100 = [results['train100'][a]['rmse'] for a in alphas_list]

ax4.plot(range(len(alphas_list)), rmse_25, 'o-', label='Train on 25', linewidth=2, markersize=8, color='blue')
ax4.plot(range(len(alphas_list)), rmse_100, 's-', label='Train on 100', linewidth=2, markersize=8, color='orange')
ax4.set_xticks(range(len(alphas_list)))
ax4.set_xticklabels([f'{a:.3f}' for a in alphas_list], rotation=45)
ax4.set_xlabel('Alpha (Regularization Parameter)', fontsize=11)
ax4.set_ylabel('RMSE on Test Set', fontsize=11)
ax4.set_title('RMSE vs Regularization Parameter', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset2\poly_regression_results.png", dpi=300, bbox_inches='tight')
print("\n✅ Plot saved as 'poly_regression_results.png'")

# -----------------------------
# 3D Surface plot of learned polynomial (plane/surface)
# -----------------------------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig3 = plt.figure(figsize=(10, 8))
ax3d = fig3.add_subplot(111, projection='3d')

# Create grid over the range of test features
x1_min, x1_max = X_test[:, 0].min(), X_test[:, 0].max()
x2_min, x2_max = X_test[:, 1].min(), X_test[:, 1].max()
x1_grid = np.linspace(x1_min, x1_max, 50)
x2_grid = np.linspace(x2_min, x2_max, 50)
xx1, xx2 = np.meshgrid(x1_grid, x2_grid)
grid_points = np.column_stack([xx1.ravel(), xx2.ravel()])

# Predict on grid (best_model is chosen earlier)
zz = best_model.predict(grid_points)
zz = zz.reshape(xx1.shape)

# Surface
surf = ax3d.plot_surface(xx1, xx2, zz, cmap='viridis', alpha=0.6, linewidth=0, antialiased=True)
fig3.colorbar(surf, shrink=0.6, aspect=10, label='Predicted output')

# Scatter actual test points
ax3d.scatter(X_test[:, 0], X_test[:, 1], y_test, color='red', s=40, label='Test points', depthshade=True)

ax3d.set_xlabel('x1')
ax3d.set_ylabel('x2')
ax3d.set_zlabel('Output')
ax3d.set_title(f'3D Surface: Best Model ({best_dataset}, α={best_alpha})')
ax3d.legend()

fig3.savefig(r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset2\poly_regression_3d.png", dpi=300, bbox_inches='tight')
print("✅ 3D surface plot saved as 'poly_regression_3d.png'")

plt.show()

# ==============================
# Summary Statistics
# ==============================
print("\n" + "="*70)
print("Summary Statistics")
print("="*70)
print(f"Best Model RMSE: {best_rmse:.6f}")
print(f"Best Model R² Score: {r2_score(y_test, best_pred):.6f}")
print(f"Mean Absolute Error: {np.mean(np.abs(residuals)):.6f}")
print(f"Standard Deviation of Residuals: {np.std(residuals):.6f}")
print(f"\nDataset used: {best_dataset}")
print(f"Training samples: {len(y100) if best_dataset == 'train100' else len(y25)}")
print(f"Test samples: {len(y_test)}")
print(f"Polynomial degree: {degree}")
print(f"Best alpha: {best_alpha}")
