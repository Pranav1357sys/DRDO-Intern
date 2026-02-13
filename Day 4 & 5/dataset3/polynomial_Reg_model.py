import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


# ------------------------------------------------
# Load all 4 files (unnamed columns)
# ------------------------------------------------
DATA_DIR = r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset3"
X_train = pd.read_csv(DATA_DIR + "\\train_data.csv", header=None).values
X_test = pd.read_csv(DATA_DIR + "\\test_data.csv", header=None).values

y_train = pd.read_csv(DATA_DIR + "\\train_label.csv", header=None).values.ravel()
y_test = pd.read_csv(DATA_DIR + "\\test_label.csv", header=None).values.ravel()

# ------------------------------------------------
# Sanity checks
# ------------------------------------------------
if X_train.shape[0] != y_train.shape[0] or X_test.shape[0] != y_test.shape[0]:
    raise ValueError("Training / test sizes do not match labels")


def safe_std_scale(X_train_raw, X_test_raw):
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std_fixed = np.where(std == 0, 1.0, std)
    Xtr = (X_train_raw - mean) / std_fixed
    Xte = (X_test_raw - mean) / std_fixed
    return Xtr, Xte


def polynomial_rmse_ridge(degree, alphas=None, cv=5):
    """Build pipeline: PolynomialFeatures -> StandardScaler -> RidgeCV
    Returns RMSE on test set and the chosen alpha.
    """
    if alphas is None:
        alphas = np.logspace(-6, 6, 25)

    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=alphas, cv=cv, scoring='neg_mean_squared_error')),
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    chosen_alpha = pipeline.named_steps['ridge'].alpha_
    return rmse, chosen_alpha, pipeline


if __name__ == '__main__':
    try:
        deg = int(input("Enter polynomial degree: "))
    except Exception:
        deg = 2

    rmse, alpha, model = polynomial_rmse_ridge(deg)
    print(f"RMSE (degree={deg}) = {rmse:.6f}    chosen alpha={alpha}")

    # Also print a simple linear baseline (degree=1) for comparison
    rmse_lin, alpha_lin, _ = polynomial_rmse_ridge(1)
    print(f"Baseline linear RMSE (degree=1) = {rmse_lin:.6f}    chosen alpha={alpha_lin}")

    # If performance is poor, suggest trying smaller degree or stronger regularization
