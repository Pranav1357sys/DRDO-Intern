import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv(r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 3\train_1b.csv")
test = pd.read_csv(r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 3\test.csv")


X_train = train["X"].values.reshape(-1,1)
y_train = train["y"].values.reshape(-1,1)

X_test = test["X"].values.reshape(-1,1)
y_test = test["y"].values.reshape(-1,1)

# ---------------------------------------
# 2. Feature normalization
# ---------------------------------------
mean = X_train.mean()
std = X_train.std()

X_train = (X_train - mean)/std
X_test = (X_test - mean)/std

# ---------------------------------------
# 3. Polynomial Degree
# ---------------------------------------
degree = 3   # Try 1,2,3,5,8

# ---------------------------------------
# 4. Design Matrix (Polynomial)
# ---------------------------------------
def design_matrix(X, degree):
    X_dm = np.ones((len(X), 1))
    for i in range(1, degree+1):
        X_dm = np.hstack((X_dm, X**i))
    return X_dm

X_train_dm = design_matrix(X_train, degree)
X_test_dm = design_matrix(X_test, degree)

# ---------------------------------------
# 5. Optimal Weights
# ---------------------------------------
W = np.linalg.inv(X_train_dm.T @ X_train_dm) @ X_train_dm.T @ y_train

# ---------------------------------------
# 6. Predictions
# ---------------------------------------
train_pred = X_train_dm @ W
test_pred = X_test_dm @ W

# ---------------------------------------
# 7. RMSE
# ---------------------------------------
train_rmse = np.sqrt(np.mean((y_train - train_pred)**2))
test_rmse = np.sqrt(np.mean((y_test - test_pred)**2))

print("Polynomial Degree:", degree)
print("Training RMSE:", train_rmse)
print("Testing RMSE:", test_rmse)

# ---------------------------------------
# 8. Combined Plot
# ---------------------------------------
plt.figure()

plt.scatter(X_train, y_train, label="Train Data")
plt.scatter(X_test, y_test, label="Test Data")

x_line = np.linspace(min(X_train), max(X_train), 200).reshape(-1,1)
x_line_dm = design_matrix(x_line, degree)
y_line = x_line_dm @ W

plt.plot(x_line, y_line, label="Polynomial Curve")

plt.title(f"Polynomial Regression (Degree = {degree})")
plt.legend()
plt.show()