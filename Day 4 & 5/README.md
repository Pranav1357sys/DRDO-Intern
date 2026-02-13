# Day 4 — Polynomial Regression (dataset2)

Today I experimented with polynomial regression models and simple model-search utilities.

## Project structure

- `2.py` — (top-level) small script / notebook runner (user project file).
- `dataset2/`
  - `poly_model_search.py` — grid/search utilities for polynomial regression experiments.
  - `Polynomial_Reg_model.py` — custom polynomial regression implementation / training script.
  - `Polynomial_Reg_sklearn.py` — polynomial regression using scikit-learn pipeline.
  - `test_design.py` — evaluation / test harness scripts.
  - `Train-2a-25.csv`, `Train-2b-100.csv`, `Val-50.csv`, `Test-50.csv` — CSV datasets used by the scripts.

## Requirements

- Python 3.8+ (3.9/3.10 recommended)
- pip packages:

```
pip install numpy pandas scikit-learn matplotlib
```

If you use a virtual environment:

```
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt  # optional if you create requirements
```

## Quick usage

- Run the scikit-learn polynomial regression example:

```
python dataset2\Polynomial_Reg_sklearn.py
```

- Run the model search / experiment script:

```
python dataset2\poly_model_search.py
```

- Run tests/evaluation harness:

```
python dataset2\test_design.py
```

Adjust dataset file paths inside the scripts if your working directory differs.

## What each file does

- `poly_model_search.py`: automates searching over polynomial degrees and hyperparameters, trains models, and prints/plots results.
- `Polynomial_Reg_model.py`: a manual/custom implementation showing how polynomial features and regression are combined.
- `Polynomial_Reg_sklearn.py`: a reproducible example using `Pipeline`, `PolynomialFeatures`, and `LinearRegression` from scikit-learn.
- `test_design.py`: small evaluation utilities and example usage for the datasets.

## Git upload checklist

Run these commands to add Day 4 content to a git repo and push (replace `<remote>` and `<branch>`):

```bash
git add README.md dataset2  
git commit -m "Add Day 4: polynomial regression examples and datasets"
git remote add origin <your-remote-url>   # only if remote not set
git push -u origin main  # or replace `main` with your branch
```

## Next steps / Notes

- Added `requirements.txt` listing core dependencies (`numpy`, `pandas`, `scikit-learn`, `matplotlib`).

  Install with:

  ```bash
  pip install -r requirements.txt
  ```

- Inserted short module docstrings into the four scripts under `dataset2/` to describe purpose and usage.

- Updated git upload checklist to include the new file and changes:

```bash
git add README.md requirements.txt dataset2
git commit -m "Add Day 4: README, requirements, and docstrings for scripts"
git remote add origin <your-remote-url>   # only if remote not set
git push -u origin main  # or replace `main` with your branch
```