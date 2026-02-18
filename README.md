# DRDO Internship - Machine Learning Diary

This repository contains the code, experiments, and learning materials from my internship at DRDO. It serves as a daily log of Machine Learning concepts I have learned and implemented, ranging from basic regression to more complex algorithms.

**Note:** This repository is a work in progress. I will be adding more content as I progress through my internship.

## 📂 Project Structure

The project is organized by days, each focusing on specific ML topics:

### [Day 1 & 2: Regularization Techniques](./Day%201%26%202)
Focuses on understanding Overfitting, Underfitting, and methods to handle them.
*   **Key Concepts:** Lasso (L1), Ridge (L2), Bias-Variance Tradeoff.
*   **Files:**
    *   `lasso regularisation.py`: Implementation of L1 regularization.
    *   `overfitting,underfitting and ridge regularisation.ipynb`: Interactive notebook exploring regularization effects.

### [Day 3: Regression Models](./Day%203)
Implementation of fundamental regression and classification algorithms.
*   **Key Concepts:** Linear Regression, Logistic Regression.
*   **Files:**
    *   `linear regression.py`
    *   `logistic regression.py`

### [Day 4 & 5: Polynomial Regression & Model Selection](./Day%204%20%26%205)
Experiments with finding the best model complexity and using Scikit-Learn pipelines.
*   **Key Concepts:** Polynomial Features, Grid Search, RMSE evaluation.
*   **Files:**
    *   `dataset2/poly_model_search.py`: Script to search for optimal polynomial degree.
    *   `dataset2/Polynomial_Reg_sklearn.py`: Scikit-learn implementation.
    *   `dataset2/test_design.py`: Evaluation harness.

### [Day 6: Algorithms from Scratch](./Day%206)
Implementing classic algorithms using only NumPy to understand the underlying mathematics.
*   **Key Concepts:** K-Nearest Neighbors (KNN), Naive Bayes.
*   **Files:**
    *   `KNN (only using numpy).py`
    *   `Naive bayes (only using numpy).py`

## 🛠️ Installation & Requirements

The project uses Python and standard data science libraries.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd DRDO-Intern
    ```

2.  **Install dependencies:**
    You can install the required packages using pip:
    ```bash
    pip install numpy pandas matplotlib scikit-learn
    ```
    *Alternatively, if a `requirements.txt` is present in a subfolder, you can use that.*

## 🚀 Usage

Navigate to the specific day's directory and run the scripts using Python or Jupyter Notebook.

**Example (Running Lasso Regression):**
```bash
cd "Day 1& 2"
python "lasso regularisation.py"
```

**Example (Running Polynomial Regression Search):**
```bash
cd "Day 4 & 5"
python dataset2/poly_model_search.py
```

## 🔜 Future Plans

*   Implementation of Support Vector Machines (SVM).
*   Decision Trees and Random Forests.
*   Neural Networks basics.
*   More comprehensive datasets and real-world examples.
*   I have started with Deep Learning now .
## 👤 Author

**Pranav Singh Rawat**
*Intern at DRDO*
