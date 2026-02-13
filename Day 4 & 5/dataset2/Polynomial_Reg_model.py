"""Polynomial_Reg_model.py

Simple, self-contained implementation for polynomial regression experiments.

Provides helpers to load the CSV datasets, build polynomial design matrices,
train a Ridge model with internal CV, evaluate RMSE/R², and visualize results.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# compact paths
P25 = r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset2\Train-2a-25.csv"
P100 = r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset2\Train-2b-100.csv"
PT = r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset2\Test-50.csv"


def load(path, sort=True):
    df = pd.read_csv(path)
    if sort:
        df = df.sort_values(["x1", "x2"]).reset_index(drop=True)
    X = df[["x1", "x2"]].values.astype(float)
    y = df["output"].values.astype(float)
    return X, y


def eval_poly(X_tr, y_tr, X_te, y_te, deg=2, alphas=None):
    poly = PolynomialFeatures(degree=deg, include_bias=True)
    Xtr = poly.fit_transform(X_tr)
    Xte = poly.transform(X_te)
    alphas = alphas if alphas is not None else np.logspace(-6, 6, 13)
    model = RidgeCV(alphas=alphas, store_cv_values=False)
    model.fit(Xtr, y_tr)
    y_pred = model.predict(Xte)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2 = r2_score(y_te, y_pred)
    return dict(poly=poly, model=model, Xtr=Xtr, Xte=Xte, y_pred=y_pred, rmse=rmse, r2=r2)


def quick_plot(X_test, y_test, infos, res=40):
    # single-row subplots
    n = len(infos)
    fig = plt.figure(figsize=(5 * n, 4))
    x1min, x1max = X_test[:, 0].min(), X_test[:, 0].max()
    x2min, x2max = X_test[:, 1].min(), X_test[:, 1].max()
    g1 = np.linspace(x1min, x1max, res)
    g2 = np.linspace(x2min, x2max, res)
    xx1, xx2 = np.meshgrid(g1, g2)
    grid = np.column_stack([xx1.ravel(), xx2.ravel()])

    for i, (name, info) in enumerate(infos, start=1):
        ax = fig.add_subplot(1, n, i, projection='3d')
        zz = info['model'].predict(info['poly'].transform(grid)).reshape(xx1.shape)
        ax.plot_surface(xx1, xx2, zz, cmap='viridis', alpha=0.6)
        ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='r', s=10)
        ax.set_title(f"{name}\nRMSE={info['rmse']:.2f} R2={info['r2']:.3f}")
    plt.tight_layout(); plt.show()


if __name__ == '__main__':
    X25, y25 = load(P25, sort=True)
    X100, y100 = load(P100, sort=True)
    Xt, yt = load(PT, sort=True)

    # quick stats
    print(f"Targets: train25 mean={y25.mean():.1f} std={y25.std():.1f}; train100 mean={y100.mean():.1f} std={y100.std():.1f}; test mean={yt.mean():.1f} std={yt.std():.1f}")

    info25 = eval_poly(X25, y25, Xt, yt, deg=2)
    info100 = eval_poly(X100, y100, Xt, yt, deg=2)

    print(f"RMSE (25->test) = {info25['rmse']:.3f}, R2 = {info25['r2']:.4f}")
    print(f"RMSE (100->test)= {info100['rmse']:.3f}, R2 = {info100['r2']:.4f}")

    # print small design-matrix sample
    print('\nDesign matrix sample (train-100 degree=2, first 5 rows):')
    print(info100['Xtr'][:5])

    quick_plot(Xt, yt, [('Train-2a-25', info25), ('Train-2b-100', info100)], res=50)