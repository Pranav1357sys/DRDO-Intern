import numpy as np
import matplotlib.pyplot as plt

# =========================================
# 1. Generate Classification Dataset
# =========================================

np.random.seed(0)

N = 100

X = np.random.randn(N,2)

# True boundary
true_w = np.array([2,-3])
b = 0.5

linear = X @ true_w + b
y = (linear > 0).astype(int)   # class labels 0 or 1

# =========================================
# 2. Add Bias Column
# =========================================

X = np.c_[np.ones(N), X]   # add column of 1s

# =========================================
# 3. Train/Test Split
# =========================================

idx = np.random.permutation(N)
split = int(0.7*N)

train = idx[:split]
test = idx[split:]

X_train = X[train]
y_train = y[train]

X_test = X[test]
y_test = y[test]

# =========================================
# 4. Sigmoid Function
# =========================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# =========================================
# 5. Logistic Regression (Gradient Descent)
# =========================================

def logistic_regression(X, y, lr=0.1, epochs=2000):

    w = np.zeros(X.shape[1])

    for _ in range(epochs):

        z = X @ w
        y_pred = sigmoid(z)

        gradient = X.T @ (y_pred - y) / len(y)

        w -= lr * gradient

    return w

# =========================================
# 6. Train Model
# =========================================

w = logistic_regression(X_train, y_train)

# =========================================
# 7. Predictions
# =========================================

prob = sigmoid(X_test @ w)
y_pred = (prob >= 0.5).astype(int)

accuracy = np.mean(y_pred == y_test)

print("Test Accuracy:", accuracy)

# =========================================
# 8. Plot Decision Boundary
# =========================================

x1 = np.linspace(-3,3,50)
x2 = -(w[0] + w[1]*x1)/w[2]

plt.scatter(X[:,1], X[:,2], c=y)
plt.plot(x1, x2)
plt.show()
