"""test_design.py

Tiny inspection utilities for dataset design-matrix checks and quick examples.

This file demonstrates constructing a polynomial design row and printing
sample values from `Train-2b-100.csv` to help verify feature engineering.
"""

import numpy as np
import pandas as pd

# Load Train100 to check
train100 = pd.read_csv(r"C:\Users\PRANAV SINGH RAWAT\OneDrive\Desktop\PRANAV\DRDO internship\Day 4\dataset2\Train-2b-100.csv")

X100 = train100[["x1","x2"]].values
y100 = train100["output"].values.reshape(-1,1)

print("First row of raw data:")
print(f"x1={X100[0,0]}, x2={X100[0,1]}, output={y100[0,0]}")

# Now let's create design matrix on RAW data (not normalized)
X1 = X100[0:1, 0:1]
X2 = X100[0:1, 1:2]

# Design matrix: [1, x1, x2, x1^2, x1*x2, x2^2]
dm = np.array([[1, X1[0,0], X2[0,0], X1[0,0]**2, X1[0,0]*X2[0,0], X2[0,0]**2]])

print("\nDesign matrix row (raw data, no normalization):")
print(f"[1, {X1[0,0]}, {X2[0,0]}, {X1[0,0]**2}, {X1[0,0]*X2[0,0]}, {X2[0,0]**2}]")
print(dm)

# Now check what polynomial should produce
# If y = w0 + w1*x1 + w2*x2 + w3*x1^2 + w4*x1*x2 + w5*x2^2
# We need to find w values that fit the data

print(f"\nTarget output: {y100[0,0]}")
print("\nThis output should come from the polynomial with unknown weights")
