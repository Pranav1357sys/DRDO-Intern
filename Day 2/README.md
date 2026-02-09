# Regularization Techniques in Machine Learning

This directory contains educational materials demonstrating various regularization techniques used in machine learning to prevent overfitting and improve model generalization.

## Files

### 1. lasso regularisation.py
A Python script that demonstrates **LASSO (L1) Regularization** using the coordinate descent algorithm.

**Key Features:**
- Dataset generation: Creates a noisy quadratic dataset
- Polynomial feature generation up to degree 8
- Train-test split (70-30)
- Implements three models:
  - **Underfit Model**: Linear regression (degree 1)
  - **Overfit Model**: High-degree polynomial (degree 8)
  - **LASSO Model**: Degree 8 with L1 regularization using coordinate descent
- Soft-thresholding mechanism for sparse weight updates
- RMSE evaluation on train and test sets
- Visualization comparing all three approaches
- Displays learned LASSO weights

**How LASSO Works:**
- Uses coordinate descent optimization
- Applies L1 penalty: encourages weight sparsity (some weights become exactly 0)
- Soft-thresholding operation shrinks weights toward zero, with smaller magnitude weights forced to zero

**Run:**

python "lasso regularisation.py"
```

---

### 2. overfitting,underfitting and ridge regularisation.ipynb
A Jupyter notebook demonstrating **Overfitting, Underfitting, and Ridge (L2) Regularization**.

**Key Sections:**
1. **Data Preparation**: Creates a noisy quadratic dataset with train-test split
2. **Helper Functions**: Polynomial feature generation and RMSE calculation
3. **Underfitting**: Linear model (degree 1) - too simple
4. **Overfitting**: High-degree polynomial (degree 10) - too complex
5. **Ridge Regression**: Degree 10 with L2 regularization (lambda = 10)
6. **Visualization & Analysis**: Compares all three approaches

**How Ridge Regression Works:**
- Uses L2 penalty (squared weights)
- Formula: $ w = (X^T X + \lambda I)^{-1} X^T y $
- Shrinks all weights proportionally (unlike LASSO)
- Prevents extreme weight values

---

## Key Concepts

### Overfitting
- Model performs well on training data but poorly on test data
- Captures noise rather than true underlying pattern
- High degree polynomial example: degree 10

### Underfitting
- Model is too simple to capture data patterns
- Poor performance on both training and test data
- Linear model example: degree 1

### Regularization
- Adds penalty term to loss function to constrain model complexity
- **L1 (LASSO)**: Penalty = $\lambda \sum |w_j|$ → Produces sparse solutions
- **L2 (Ridge)**: Penalty = $\lambda \sum w_j^2$ → Shrinks all weights

---

## Prerequisites

```
numpy
matplotlib
```

Install dependencies:

pip install numpy matplotlib
```

For the Jupyter notebook, also install:

pip install jupyter
```

---

## Learning Objectives

After working through these files, you should understand:
- ✓ The differences between overfitting and underfitting
- ✓ How regularization techniques prevent overfitting
- ✓ The difference between L1 (LASSO) and L2 (Ridge) regularization
- ✓ Coordinate descent optimization for LASSO
- ✓ How to evaluate models using train/test RMSE

---

## Author Notes

These are educational implementations demonstrating fundamental ML concepts. In production, use high-level libraries like scikit-learn for regularized regression:

```python
from sklearn.linear_model import Lasso, Ridge

# LASSO
lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)

# Ridge
ridge = Ridge(alpha=10.0)
ridge.fit(X_train, y_train)
```

---

## Context

Focus: Understanding regularization techniques and their practical applications
