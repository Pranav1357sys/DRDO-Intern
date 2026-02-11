import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# ================= LOAD FILES =================
train25 = pd.read_csv(r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset2\Train-2a-25.csv")
train100 = pd.read_csv(r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset2\Train-2b-100.csv")
test = pd.read_csv(r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset2\Test-50.csv")

# ================= SORT DATASETS =================
train25_sorted = train25.sort_values(by=train25.columns[0]).reset_index(drop=True)
train100_sorted = train100.sort_values(by=train100.columns[0]).reset_index(drop=True)
test_sorted = test.sort_values(by=test.columns[0]).reset_index(drop=True)

X25 = train25_sorted.iloc[:, :-1].values
y25 = train25_sorted.iloc[:, -1].values

X100 = train100_sorted.iloc[:, :-1].values
y100 = train100_sorted.iloc[:, -1].values

Xtest = test_sorted.iloc[:, :-1].values
ytest = test_sorted.iloc[:, -1].values

print("\nDatasets sorted by first column!")

# Hardcoding degree to 2 for demonstration
degree = 2
print(f"Using degree: {degree}")

# ================= DESIGN MATRIX (RAW – FOR CHECKING ONLY) =================
poly = PolynomialFeatures(degree=degree, include_bias=True)

Phi_raw = poly.fit_transform(X25)

print("\nRAW Design Matrix (first row):")
print(Phi_raw[0])
print("Columns:", Phi_raw.shape[1])

# ================= SCALE =================
scaler = StandardScaler()

X100_s = scaler.fit_transform(X100)
X25_s = scaler.transform(X25)
Xtest_s = scaler.transform(Xtest)

# ================= POLY AFTER SCALING =================
Phi100 = poly.fit_transform(X100_s)
Phi25 = poly.transform(X25_s)
Phi_test = poly.transform(Xtest_s)

# ================= TRAIN =================
model = LinearRegression()

model.fit(Phi25, y25)
pred1 = model.predict(Phi_test)
rmse25 = np.sqrt(mean_squared_error(ytest, pred1))

model.fit(Phi100, y100)
pred2 = model.predict(Phi_test)
rmse100 = np.sqrt(mean_squared_error(ytest, pred2))

print("\nRMSE Train25:", rmse25)
print("RMSE Train100:", rmse100)

if rmse25 < rmse100:
    print("\nTrain25 better")
else:
    print("\nTrain100 better")