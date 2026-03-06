# Machine Learning Algorithms Implementation

A lightweight educational project implementing fundamental machine learning classification algorithms from scratch using only NumPy.

## 📋 Overview

This repository contains clean, well-documented implementations of two essential machine learning algorithms:
- **K-Nearest Neighbors (KNN)** - A non-parametric, instance-based learning algorithm
- **Naive Bayes** - A probabilistic classifier based on Bayes' theorem

Both implementations use only NumPy to avoid dependencies on high-level ML libraries, making them perfect for understanding the mathematical foundations of these algorithms.

## 🎯 What's Inside

### 1. KNN (K-Nearest Neighbors)
**File:** `KNN (only using numpy).py`

A simple classification algorithm that predicts the class of new data points based on the classes of their k nearest neighbors in the training set.

**How it works:**
1. Takes a test point and calculates its Euclidean distance to all training points
2. Sorts these distances and selects the k nearest points
3. Uses majority voting among these k neighbors to determine the class

**Key parameters:**
- **k** - Number of neighbors to consider (user input)

**Use cases:**
- Handwriting recognition
- Image classification
- Recommendation systems
- Simple classification tasks with small datasets

---

### 2. Naive Bayes
**File:** `Naive bayes (only using numpy).py`

A probabilistic classifier that applies Bayes' theorem with a "naive" assumption that features are independent.

**How it works:**
1. Calculates mean and variance of features for each class during training
2. Computes prior probability (P(class)) and likelihood using Gaussian distribution
3. Selects the class with the highest posterior probability

**Key advantages:**
- Fast training and prediction
- Works well with small datasets
- Probabilistic approach provides confidence estimates

**Use cases:**
- Spam email detection
- Text classification
- Sentiment analysis
- Medical diagnosis support

---

## 🚀 Getting Started

### Requirements
```
Python 3.x
NumPy
```

### Installation
```bash
pip install numpy
```

### Usage

#### Running KNN Classifier
```bash
python "KNN (only using numpy).py"
```
The script will prompt you to enter the value of k (number of neighbors).

#### Running Naive Bayes Classifier
```bash
python "Naive bayes (only using numpy).py"
```

### Example Output
Both scripts use sample 2D datasets for demonstration:
- **Training Dataset:** 6 points classified into 2 classes
- **Test Dataset:** 2 new points to classify

The scripts will output predicted class labels for the test points.

---

## 📊 Code Structure

Both implementations follow a similar class-based structure for easy understanding:

```python
class Algorithm:
    def fit(self, X, y):
        """Train the model on training data"""
        pass
    
    def predict(self, X):
        """Make predictions on new data"""
        pass
```

---

## 💡 Key Concepts Explained

### Euclidean Distance (KNN)
$$\text{distance} = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

### Gaussian/Normal Distribution (Naive Bayes)
$$P(x|c) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

### Bayes' Theorem
$$P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}$$

---

## 🔧 Customization

To use your own datasets, modify the `X_train`, `y_train`, and `X_test` arrays in either script:

```python
# Replace with your data
X_train = np.array([
    [feature1, feature2],
    [feature1, feature2],
    ...
])

y_train = np.array([class1, class2, ...])
```

---

## 📚 Educational Value

These implementations are ideal for:
- ✅ Understanding ML algorithms from first principles
- ✅ Learning NumPy array operations and linear algebra
- ✅ Preparing for machine learning interviews
- ✅ Teaching ML concepts without framework abstractions

---

## 🎓 Context

Developed as part of the DRDO Internship program (Day 6), focusing on building ML fundamentals from scratch.

---

## 📝 License

This project is open-source and available for educational purposes.

---

## 💬 Notes

- For production use, consider using scikit-learn which provides optimized implementations
- These algorithms can be extended to multi-class classification problems
- Performance can be improved with proper feature scaling and dimensionality reduction
- Experiment with different k values in KNN to see how it affects results

---

**Happy Learning!** 🎯
