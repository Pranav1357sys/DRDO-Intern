# Gradient Descent Implementation

This folder contains a simple Python implementation of the **Gradient Descent algorithm**.  
The purpose of this script is to demonstrate how model parameters can be optimized by repeatedly updating them in the direction that reduces the loss function.

Gradient descent is one of the most important optimization techniques used in machine learning and deep learning. It is commonly used when training models such as linear regression, logistic regression, and neural networks.

---

## About the Script

The `gradient_descent.py` file implements gradient descent from scratch to help understand how the algorithm works step by step.

Instead of relying on high-level machine learning libraries, the implementation focuses on the core mathematical idea behind parameter optimization.

The script demonstrates:

- How gradients are computed
- How parameters are updated iteratively
- How the loss changes during optimization
- How gradient descent moves towards the minimum of a loss function

This makes it easier to build an intuitive understanding of optimization in machine learning.

---

## How Gradient Descent Works

At a high level, gradient descent works by repeatedly adjusting model parameters in order to reduce the error between predicted values and actual values.

The parameters are updated using the rule:

θ = θ − α ∇J(θ)

Where:

- **θ** represents the model parameters  
- **α** is the learning rate (step size)  
- **∇J(θ)** is the gradient of the loss function  

By continuously applying this update rule, the algorithm gradually moves toward the minimum of the loss function.

---

## File Structure

```
Gradient Descent/
│
└── gradient_descent.py
```

---

## Requirements

To run the script, install the following Python libraries:

```
pip install numpy matplotlib
```

---

## Running the Code

Clone the repository and navigate to this folder:

```
git clone https://github.com/Pranav1357sys/DRDO-Intern.git
cd DRDO-Intern/Gradient\ Descent
```

Then run the script:

```
python gradient_descent.py
```

---

## Purpose of This Implementation

This implementation was created mainly for learning and experimentation. Writing gradient descent from scratch helps in understanding:

- The mathematical intuition behind optimization
- The role of the learning rate
- How parameter updates affect loss
- How machine learning models actually learn during training

---

## Author

Pranav Rawat  
Machine Learning and Data Science Enthusiast  