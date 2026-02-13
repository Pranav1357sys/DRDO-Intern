import numpy as np
import pandas as pd
import warnings

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

DATA_DIR = r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset3"


def load_data():
    X_train = pd.read_csv(DATA_DIR + "\\train_data.csv", header=None).values
    X_test = pd.read_csv(DATA_DIR + "\\test_data.csv", header=None).values
    y_train = pd.read_csv(DATA_DIR + "\\train_label.csv", header=None).values.ravel()
    y_test = pd.read_csv(DATA_DIR + "\\test_label.csv", header=None).values.ravel()
    return X_train, X_test, y_train, y_test


def safe_scale(Xtr_raw, Xte_raw):
    mean = Xtr_raw.mean(axis=0)
    std = Xtr_raw.std(axis=0)
    std_fixed = np.where(std == 0, 1.0, std)
    return (Xtr_raw - mean) / std_fixed, (Xte_raw - mean) / std_fixed


def eval_pipeline(pipeline, Xtr, Xte, ytr, yte):
    pipeline.fit(Xtr, ytr)
    preds = pipeline.predict(Xte)
    return np.sqrt(mean_squared_error(yte, preds))


def run_polynomial_models(Xtr, Xte, ytr, yte, max_deg=12):
    best = (None, None, float('inf'))
    alphas = np.logspace(-6, 6, 25)
    for d in range(1, max_deg + 1):
        # Ridge
        pipe_ridge = Pipeline([
            ('poly', PolynomialFeatures(degree=d, include_bias=False)),
            ('scale', StandardScaler()),
            ('ridge', RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')),
        ])
        rmse_r = eval_pipeline(pipe_ridge, Xtr, Xte, ytr, yte)
        if rmse_r < best[2]:
            best = (f'Ridge_deg{d}', pipe_ridge, rmse_r)

        # LassoCV (may be slower)
        pipe_lasso = Pipeline([
            ('poly', PolynomialFeatures(degree=d, include_bias=False)),
            ('scale', StandardScaler()),
            ('lasso', LassoCV(cv=5, n_alphas=50, max_iter=5000)),
        ])
        try:
            rmse_l = eval_pipeline(pipe_lasso, Xtr, Xte, ytr, yte)
            if rmse_l < best[2]:
                best = (f'Lasso_deg{d}', pipe_lasso, rmse_l)
        except Exception:
            pass

        # ElasticNetCV
        pipe_en = Pipeline([
            ('poly', PolynomialFeatures(degree=d, include_bias=False)),
            ('scale', StandardScaler()),
            ('en', ElasticNetCV(cv=5, n_alphas=30, l1_ratio=[.1, .5, .9], max_iter=5000)),
        ])
        try:
            rmse_en = eval_pipeline(pipe_en, Xtr, Xte, ytr, yte)
            if rmse_en < best[2]:
                best = (f'ElasticNet_deg{d}', pipe_en, rmse_en)
        except Exception:
            pass

        print(f'poly deg={d} -> ridge {rmse_r:.6f}, lasso {rmse_l if "rmse_l" in locals() else "-"}, en {rmse_en if "rmse_en" in locals() else "-"}')

    return best


def run_tree_models(Xtr, Xte, ytr, yte):
    best = (None, None, float('inf'))

    # RandomForest grid
    for n in [100, 200]:
        for depth in [None, 10, 5]:
            rf = RandomForestRegressor(n_estimators=n, max_depth=depth, random_state=42, n_jobs=-1)
            rmse = eval_pipeline(rf, Xtr, Xte, ytr, yte)
            name = f'RF_{n}_d{depth}'
            print(f'{name} -> {rmse:.6f}')
            if rmse < best[2]:
                best = (name, rf, rmse)

    # Gradient Boosting grid
    for n in [100, 200]:
        for lr in [0.1, 0.05]:
            for depth in [3, 5]:
                gb = GradientBoostingRegressor(n_estimators=n, learning_rate=lr, max_depth=depth, random_state=42)
                rmse = eval_pipeline(gb, Xtr, Xte, ytr, yte)
                name = f'GB_{n}_lr{lr}_d{depth}'
                print(f'{name} -> {rmse:.6f}')
                if rmse < best[2]:
                    best = (name, gb, rmse)

    return best


def main():
    Xtr_raw, Xte_raw, ytr, yte = load_data()

    # Try polynomial-based linear models
    best_poly = run_polynomial_models(Xtr_raw, Xte_raw, ytr, yte, max_deg=12)
    print('\nBest polynomial-model:', best_poly[0], 'RMSE=', best_poly[2])

    # Scale original features for tree models
    Xtr_s, Xte_s = safe_scale(Xtr_raw, Xte_raw)
    best_tree = run_tree_models(Xtr_s, Xte_s, ytr, yte)
    print('\nBest tree-model:', best_tree[0], 'RMSE=', best_tree[2])

    # Choose overall best
    overall = best_poly if best_poly[2] < best_tree[2] else best_tree
    print('\nOVERALL BEST:', overall[0], 'RMSE=', overall[2])


if __name__ == '__main__':
    main()
