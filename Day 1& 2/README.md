# Regularization Techniques in Machine Learning

## Overview
This folder contains materials and implementations for **Day 1 & 2** of the DRDO internship, focusing on regularization techniques in machine learning. Regularization is essential for preventing overfitting and improving model generalization.

## Contents

### 1. **Lasso Regularisation** (`lasso regularisation.py`)
Implementation of Lasso (L1) regularization for linear regression.

**Key Concepts:**
- **L1 Regularization (Lasso)**: Adds penalty term proportional to absolute value of weights
- Cost function: `J(w) = MSE(w) + λ * Σ|w_i|`
- Feature selection: Forces some weights to exactly zero
- Useful for sparse models and feature importance

**What it does:**
- Trains linear regression models with L1 penalty
- Demonstrates how Lasso shrinks coefficients to zero
- Shows sparse feature selection in action

**Usage:**
```bash
python "lasso regularisation.py"
```

### 2. **Overfitting, Underfitting & Ridge Regularisation** (`overfitting,underfitting and ridge regularisation.ipynb`)
Interactive Jupyter Notebook exploring regularization concepts and Ridge (L2) regression.

**Key Topics:**
- **Overfitting**: Model fits training data too well, poor generalization
- **Underfitting**: Model too simple, high bias, high error
- **Bias-Variance Tradeoff**: Balancing model complexity
- **Ridge Regression (L2)**: Adds penalty proportional to squared weights
  - Cost function: `J(w) = MSE(w) + λ * Σ(w_i)²`
  - Shrinks large weights without forcing them to zero
  - Better numerical stability than standard least squares

**Features:**
- Visual demonstrations of overfitting vs underfitting
- Comparison of different regularization strengths
- Ridge regression implementation and analysis
- Learning curves and cross-validation examples

**Usage:**
```bash
jupyter notebook "overfitting,underfitting and ridge regularisation.ipynb"
```

## Regularization Comparison

| Technique | Penalty | Effect | Use Case |
|-----------|---------|--------|----------|
| **Lasso (L1)** | λ·Σ\|w_i\| | Sparse (zeros out features) | Feature selection |
| **Ridge (L2)** | λ·Σ(w_i)² | Shrinks all weights | Stability, correlated features |
| **Elastic Net** | λ₁·Σ\|w_i\| + λ₂·Σ(w_i)² | Combination | General purpose |

## Key Learnings

1. **Regularization Parameter (λ)**:
   - λ = 0: No regularization (can overfit)
   - λ → ∞: All weights → 0 (underfitting)
   - Sweet spot: Found via cross-validation

2. **When to use each:**
   - **Lasso**: When you suspect only few features matter
   - **Ridge**: When all features are important, small weights preferred
   - **Elastic Net**: When you want both benefits

3. **Practical Tips**:
   - Always normalize/standardize features before regularization
   - Use cross-validation to tune λ
   - Monitor both training and validation errors

## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib (for visualizations)
- Jupyter (for notebook)

## Installation
```bash
pip install numpy pandas matplotlib jupyter scikit-learn
```

## Files Structure
```
Day 1& 2/
├── lasso regularisation.py                              # L1 regularization implementation
├── overfitting,underfitting and ridge regularisation.ipynb  # Ridge regression & concepts
└── README.md                                             # This file
```

## References
- Regularization reduces model complexity
- L1 (Lasso) for sparse solutions
- L2 (Ridge) for stable solutions
- Elasticsearch Net combines both approaches
