import numpy as np
import matplotlib.pyplot as plt

# ======================================================
# 1. Create Dataset
# ======================================================

np.random.seed(0)

x = np.linspace(0, 10, 40)
y = x**2 + np.random.randn(40)*20   # noisy quadratic

# Normalize x (important)
x = (x - np.mean(x)) / np.std(x)
x = x.reshape(-1,1)

# ======================================================
# 2. Train/Test Split
# ======================================================

idx = np.random.permutation(len(x))
split = int(0.7 * len(x))

train = idx[:split]
test = idx[split:]

X_train = x[train]
y_train = y[train]

X_test = x[test]
y_test = y[test]

# ======================================================
# 3. Polynomial Features
# ======================================================

def poly_features(x, degree):
    return np.column_stack([x.flatten()**i for i in range(degree+1)])

# Degrees
under_degree = 1      # underfit
over_degree = 8       # overfit
lasso_degree = 8      # same as overfit

Phi_train_under = poly_features(X_train, under_degree)
Phi_test_under = poly_features(X_test, under_degree)
Phi_all_under = poly_features(x, under_degree)

Phi_train_over = poly_features(X_train, over_degree)
Phi_test_over = poly_features(X_test, over_degree)
Phi_all_over = poly_features(x, over_degree)

Phi_train_lasso = poly_features(X_train, lasso_degree)
Phi_test_lasso = poly_features(X_test, lasso_degree)
Phi_all_lasso = poly_features(x, lasso_degree)

# ======================================================
# 4. RMSE
# ======================================================

def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat)**2))

# ======================================================
# 5. UNDERFIT (Linear Regression)
# ======================================================

w_under = np.linalg.pinv(Phi_train_under) @ y_train

y_under_train = Phi_train_under @ w_under
y_under_test = Phi_test_under @ w_under
y_under_all = Phi_all_under @ w_under

# ======================================================
# 6. OVERFIT (High Degree Polynomial)
# ======================================================

w_over = np.linalg.pinv(Phi_train_over) @ y_train

y_over_train = Phi_train_over @ w_over
y_over_test = Phi_test_over @ w_over
y_over_all = Phi_all_over @ w_over

# ======================================================
# 7. LASSO (Coordinate Descent)
# ======================================================

def soft_threshold(z, lam):
    if z > lam:
        return z - lam
    elif z < -lam:
        return z + lam
    else:
        return 0.0

def lasso(Phi, y, lam=0.01, epochs=2000):

    n, d = Phi.shape
    w = np.zeros(d)

    for _ in range(epochs):
        for j in range(d):

            y_pred = Phi @ w - Phi[:,j]*w[j]
            rho = Phi[:,j].T @ (y - y_pred)
            z = np.sum(Phi[:,j]**2)

            w[j] = soft_threshold(rho/z, lam)

    return w

w_lasso = lasso(Phi_train_lasso, y_train, lam=0.01)

y_lasso_train = Phi_train_lasso @ w_lasso
y_lasso_test = Phi_test_lasso @ w_lasso
y_lasso_all = Phi_all_lasso @ w_lasso

# ======================================================
# 8. Print RMSE
# ======================================================

print("\nTRAIN RMSE")
print("Underfit:", rmse(y_train, y_under_train))
print("Overfit :", rmse(y_train, y_over_train))
print("Lasso   :", rmse(y_train, y_lasso_train))

print("\nTEST RMSE")
print("Underfit:", rmse(y_test, y_under_test))
print("Overfit :", rmse(y_test, y_over_test))
print("Lasso   :", rmse(y_test, y_lasso_test))

# ======================================================
# 9. Plot Everything
# ======================================================

order = np.argsort(x.flatten())

plt.scatter(x[order], y[order], label="Data")

plt.plot(x[order], y_under_all[order], label="Underfit", linewidth=2)
plt.plot(x[order], y_over_all[order], label="Overfit", linewidth=2)
plt.plot(x[order], y_lasso_all[order], label="Lasso", linewidth=2)

plt.legend()
plt.show()

# ======================================================
# 10. Show Lasso Weights
# ======================================================

print("\nLasso Weights:")
print(w_lasso)
