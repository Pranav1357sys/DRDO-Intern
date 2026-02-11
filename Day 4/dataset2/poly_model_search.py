"""poly_model_search.py

Utilities to run grid/search experiments for polynomial regression models.

This script loads provided CSV datasets, defines candidate pipelines and
search grids, runs GridSearchCV for several estimators, and saves the
best model predictions and a 3D surface visualization.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


# load
def load(path):
    df = pd.read_csv(path)
    return df[["x1", "x2"]].values.astype(float), df["output"].values.ravel().astype(float)

X25, y25 = load(r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset2\Train-2a-25.csv")
X100, y100 = load(r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset2\Train-2b-100.csv")
X_test, y_test = load(r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset2\Test-50.csv")

X_train, y_train = X100, y100

print(f"Data shapes: train={X_train.shape}, test={X_test.shape}")


# candidate models and grids
from functools import partial
pipe = lambda model: Pipeline([('poly', PolynomialFeatures(include_bias=True)), ('scaler', StandardScaler()), ('model', model)])

search_space = {
    'Ridge': (pipe(Ridge()), {'poly__degree': [1, 2, 3], 'model__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]}),
    'Huber': (pipe(HuberRegressor()), {'poly__degree': [1, 2], 'model__epsilon': [1.1, 1.35], 'model__alpha':[1e-4,1e-3,1e-2]}),
    'RandomForest': (pipe(RandomForestRegressor(random_state=42)), {'poly__degree':[1], 'model__n_estimators':[100,300], 'model__max_depth':[3,6]}),
    'GradientBoosting': (pipe(GradientBoostingRegressor(random_state=42)), {'poly__degree':[1,2], 'model__n_estimators':[100,300], 'model__learning_rate':[0.01,0.1]})
}


def run_search(name, estimator, grid):
    gs = GridSearchCV(estimator, grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    cv_rmse = np.sqrt(-gs.best_score_)
    pred = best.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name:15s} | CV RMSE: {cv_rmse:8.4f} | Test RMSE: {test_rmse:8.4f}")
    return {'name': name, 'est': best, 'cv_rmse': cv_rmse, 'test_rmse': test_rmse, 'pred': pred, 'gs': gs}


results = [run_search(n, *search_space[n]) for n in search_space]

# pick best
best = min(results, key=lambda r: r['test_rmse'])
print('\nBest overall:', best['name'], 'Test RMSE=', f"{best['test_rmse']:.4f}")

# save predictions
out = pd.DataFrame({'x1': X_test[:,0], 'x2': X_test[:,1], 'actual': y_test, 'pred': best['pred']})
out['error'] = out['actual'] - out['pred']
out.to_csv('best_model_test_predictions.csv', index=False)

# 3d surface
from mpl_toolkits.mplot3d import Axes3D  # noqa
fig = plt.figure(figsize=(9,7)); ax = fig.add_subplot(111, projection='3d')
g1,g2 = np.linspace(X_test[:,0].min(), X_test[:,0].max(), 60), np.linspace(X_test[:,1].min(), X_test[:,1].max(), 60)
xx,yy = np.meshgrid(g1,g2); grid = np.column_stack([xx.ravel(), yy.ravel()])
zz = best['est'].predict(grid).reshape(xx.shape)
ax.plot_surface(xx, yy, zz, cmap='coolwarm', alpha=0.7)
ax.scatter(X_test[:,0], X_test[:,1], y_test, color='k', s=25)
ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('output')
plt.tight_layout(); plt.savefig('best_model_3d_surface.png', dpi=300, bbox_inches='tight')
print("Saved best_model_3d_surface.png")
plt.show()

print('\nDone.')
