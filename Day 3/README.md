## Project Overview

Today I learned about linear and logistic regression models, implemented code for them ,i.e trained and tested on CSV datasets.

## Regression Models

### Linear Regression

Linear regression is a supervised learning algorithm used for predicting continuous numerical values. It assumes a linear relationship between input features and the target variable. The model fits a straight line (or hyperplane in multi-dimensional space) through the data points to minimize the prediction error.

**Key Points:**
- Used for regression problems (continuous output)
- Assumes linear relationship between features and target
- Minimizes Mean Squared Error (MSE)
- Output: Continuous values

### Logistic Regression

Logistic regression is a supervised learning algorithm used for binary classification problems. Despite its name, it's a classification algorithm that uses the logistic (sigmoid) function to model the probability of a binary outcome. It predicts the probability that an instance belongs to a particular class.

**Key Points:**
- Used for classification problems (discrete/binary output)
- Outputs probability values between 0 and 1
- Uses sigmoid activation function
- Minimizes log-loss (cross-entropy)
- Output: Probability/Class prediction (0 or 1)

## Files

- **linear regression.py** - Implementation of linear regression model
- **logistic regression.py** - Implementation of logistic regression model
- **train_1b.csv** - Training dataset
- **test.csv** - Test dataset for model evaluation

## Requirements

- Python 3.x
- pandas
- scikit-learn
- numpy
- matplotlib (for visualization, if used)

## Installation

Install required packages using pip:

```bash
pip install pandas scikit-learn numpy matplotlib
```

## Usage

Run the regression models:

```bash
python linear regression.py
python logistic regression.py
```

## Dataset

- **Training Data**: `train_1b.csv`
- **Test Data**: `test.csv`

Ensure both CSV files are in the same directory as the Python scripts.

## Author

DRDO Internship - Day 3

## License

[Add your license information here]
