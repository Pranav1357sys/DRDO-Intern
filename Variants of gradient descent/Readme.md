# Variants of Gradient Descent

This folder contains implementations of different variants of the **Gradient Descent optimization algorithm** using Python.  
The purpose of this project is to understand how different optimization techniques behave while minimizing a loss function and how they improve the learning process in machine learning models.

Gradient descent is one of the core concepts in machine learning and deep learning, and different variants of it are designed to improve convergence speed, stability, and overall optimization performance.

---

## About the Project

The implementations in this folder focus on comparing multiple gradient descent techniques and understanding their differences through practical experimentation.

The project demonstrates:

- How optimization algorithms update parameters
- The effect of learning rate on convergence
- Differences between batch, stochastic, and mini-batch approaches
- Faster and more stable optimization using advanced techniques

This project was mainly created for learning purposes to build a stronger intuition about optimization algorithms used in machine learning.

---

## Variants Implemented

### 1. Batch Gradient Descent
Uses the complete dataset to calculate gradients before updating parameters.

**Characteristics:**
- Stable convergence
- Computationally expensive for large datasets
- Slower updates

---

### 2. Stochastic Gradient Descent (SGD)
Updates parameters using only one training example at a time.

**Characteristics:**
- Faster updates
- More noisy optimization path
- Can escape local minima more easily

---

### 3. Mini-Batch Gradient Descent
Uses a small subset of the dataset for parameter updates.

**Characteristics:**
- Balance between speed and stability
- Commonly used in deep learning
- More efficient than batch gradient descent

---

### 4. Momentum Gradient Descent
Adds momentum to parameter updates to accelerate convergence.

**Characteristics:**
- Faster convergence
- Reduces oscillations
- Helps optimization move smoothly through the loss landscape

---

### 5. Adam Optimizer
Combines momentum and adaptive learning rates.

**Characteristics:**
- Faster and efficient optimization
- Widely used in neural networks
- Handles sparse gradients effectively

---

## Folder Structure

``` id="6af4np"
Variants of Gradient Descent/
│
├── batch_gradient_descent.py
├── stochastic_gradient_descent.py
├── mini_batch_gradient_descent.py
├── momentum_gradient_descent.py
├── adam_optimizer.py
│
└── README.md
```

---

## Requirements

Install the required libraries before running the scripts:

``` id="a4om8j"
pip install numpy matplotlib
```

---

## Running the Code

Clone the repository and navigate to this folder:

``` id="b0t75w"
git clone https://github.com/Pranav1357sys/DRDO-Intern.git
cd DRDO-Intern/Variants\ of\ Gradient\ Descent
```

Run any implementation using:

``` id="26n3r3"
python filename.py
```

Example:

``` id="9bjz4f"
python stochastic_gradient_descent.py
```

---

## What I Learned From This Project

While implementing these algorithms, I gained a better understanding of:

- How optimization works internally in machine learning
- Why different gradient descent variants exist
- The importance of learning rates
- How parameter updates affect convergence
- Why optimizers like Adam are widely used in deep learning

Building these implementations from scratch helped me understand the mathematical intuition behind optimization instead of just using built-in libraries.

---

## Applications

Gradient descent variants are widely used in:

- Linear Regression
- Logistic Regression
- Neural Networks
- Deep Learning Models
- Computer Vision
- Natural Language Processing

---

## Future Improvements

Some future improvements for this project include:

- Adding RMSProp and AdaGrad implementations
- Visualizing optimization paths using contour plots
- Comparing convergence rates graphically
- Applying these optimizers to neural network training

---

## Author

Pranav Singh Rawat  
Machine Learning and Data Science Enthusiast  

GitHub:  
https://github.com/Pranav1357sys