"""
Gradient Descent Variants Implementation
Compares multiple variants: BGD, SGD, Mini-batch GD, Momentum, RMSprop, and Adam
Automatically calculates accuracy and visualizes descent paths on a contour map
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# ==================== DATA GENERATION ====================
def generate_data(n_samples=100, random_state=42):
    """Generate synthetic dataset for regression"""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, 1) * 2
    # True relationship: y = 3*x + 2 + noise
    Y = 3 * X + 2 + np.random.randn(n_samples, 1) * 0.5
    return X, Y


# ==================== LOSS FUNCTION ====================
def compute_loss(X, Y, w, b):
    """Compute Mean Squared Error"""
    m = len(X)
    predictions = X * w + b
    loss = np.sum((predictions - Y) ** 2) / (2 * m)
    return loss


def compute_gradients(X, Y, w, b):
    """Compute gradients of loss w.r.t w and b"""
    m = len(X)
    predictions = X * w + b
    dw = np.sum((predictions - Y) * X) / m
    db = np.sum(predictions - Y) / m
    return dw, db


def compute_accuracy(Y_true, Y_pred):
    """Compute R-squared score and RMSE"""
    ss_res = np.sum((Y_true - Y_pred) ** 2)
    ss_tot = np.sum((Y_true - np.mean(Y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    rmse = np.sqrt(np.mean((Y_true - Y_pred) ** 2))
    return r_squared, rmse


# ==================== GRADIENT DESCENT VARIANTS ====================

class GradientDescentVariant:
    """Base class for gradient descent variants"""
    def __init__(self, X, Y, lr=0.01, epochs=500, name="GD"):
        self.X = X
        self.Y = Y
        self.lr = lr
        self.epochs = epochs
        self.name = name
        self.w_history = []
        self.b_history = []
        self.loss_history = []
        
    def train(self):
        raise NotImplementedError
        
    def get_metrics(self):
        """Calculate final accuracy metrics"""
        predictions = self.X * self.w + self.b
        r2, rmse = compute_accuracy(self.Y, predictions)
        return {
            'name': self.name,
            'w': self.w,
            'b': self.b,
            'r2_score': r2,
            'rmse': rmse,
            'final_loss': self.loss_history[-1]
        }


class BatchGradientDescent(GradientDescentVariant):
    """Batch Gradient Descent - uses all samples in each iteration"""
    def train(self):
        self.w, self.b = -2.0, -2.0
        for epoch in range(self.epochs):
            dw, db = compute_gradients(self.X, self.Y, self.w, self.b)
            self.w -= self.lr * dw
            self.b -= self.lr * db
            loss = compute_loss(self.X, self.Y, self.w, self.b)
            self.loss_history.append(loss)
            self.w_history.append(self.w)
            self.b_history.append(self.b)
        return self


class StochasticGradientDescent(GradientDescentVariant):
    """Stochastic Gradient Descent - uses one sample at each iteration"""
    def train(self):
        self.w, self.b = -2.0, -2.0
        m = len(self.X)
        for epoch in range(self.epochs):
            indices = np.random.permutation(m)
            for idx in indices:
                x_i = self.X[idx:idx+1]
                y_i = self.Y[idx:idx+1]
                dw, db = compute_gradients(x_i, y_i, self.w, self.b)
                self.w -= self.lr * dw
                self.b -= self.lr * db
            loss = compute_loss(self.X, self.Y, self.w, self.b)
            self.loss_history.append(loss)
            self.w_history.append(self.w)
            self.b_history.append(self.b)
        return self


class MiniBatchGradientDescent(GradientDescentVariant):
    """Mini-batch Gradient Descent - uses batch_size samples at each iteration"""
    def __init__(self, X, Y, lr=0.01, epochs=500, batch_size=16, name="Mini-batch GD"):
        super().__init__(X, Y, lr, epochs, name)
        self.batch_size = batch_size
        
    def train(self):
        self.w, self.b = -2.0, -2.0
        m = len(self.X)
        for epoch in range(self.epochs):
            indices = np.random.permutation(m)
            for i in range(0, m, self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                X_batch = self.X[batch_indices]
                Y_batch = self.Y[batch_indices]
                dw, db = compute_gradients(X_batch, Y_batch, self.w, self.b)
                self.w -= self.lr * dw
                self.b -= self.lr * db
            loss = compute_loss(self.X, self.Y, self.w, self.b)
            self.loss_history.append(loss)
            self.w_history.append(self.w)
            self.b_history.append(self.b)
        return self


class MomentumGradientDescent(GradientDescentVariant):
    """Gradient Descent with Momentum"""
    def __init__(self, X, Y, lr=0.01, epochs=500, momentum=0.9, name="Momentum"):
        super().__init__(X, Y, lr, epochs, name)
        self.momentum = momentum
        
    def train(self):
        self.w, self.b = -2.0, -2.0
        vw, vb = 0, 0
        for epoch in range(self.epochs):
            dw, db = compute_gradients(self.X, self.Y, self.w, self.b)
            vw = self.momentum * vw - self.lr * dw
            vb = self.momentum * vb - self.lr * db
            self.w += vw
            self.b += vb
            loss = compute_loss(self.X, self.Y, self.w, self.b)
            self.loss_history.append(loss)
            self.w_history.append(self.w)
            self.b_history.append(self.b)
        return self


class RMSpropGradientDescent(GradientDescentVariant):
    """RMSprop - Root Mean Square Propagation"""
    def __init__(self, X, Y, lr=0.01, epochs=500, beta=0.999, epsilon=1e-8, name="RMSprop"):
        super().__init__(X, Y, lr, epochs, name)
        self.beta = beta
        self.epsilon = epsilon
        
    def train(self):
        self.w, self.b = -2.0, -2.0
        sw, sb = 0, 0
        for epoch in range(self.epochs):
            dw, db = compute_gradients(self.X, self.Y, self.w, self.b)
            sw = self.beta * sw + (1 - self.beta) * dw**2
            sb = self.beta * sb + (1 - self.beta) * db**2
            self.w -= self.lr * dw / (np.sqrt(sw) + self.epsilon)
            self.b -= self.lr * db / (np.sqrt(sb) + self.epsilon)
            loss = compute_loss(self.X, self.Y, self.w, self.b)
            self.loss_history.append(loss)
            self.w_history.append(self.w)
            self.b_history.append(self.b)
        return self


class AdamGradientDescent(GradientDescentVariant):
    """Adam - Adaptive Moment Estimation"""
    def __init__(self, X, Y, lr=0.01, epochs=500, beta1=0.9, beta2=0.999, epsilon=1e-8, name="Adam"):
        super().__init__(X, Y, lr, epochs, name)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
    def train(self):
        self.w, self.b = -2.0, -2.0
        mw, mb = 0, 0  # First moment
        vw, vb = 0, 0  # Second moment
        t = 0
        for epoch in range(self.epochs):
            t += 1
            dw, db = compute_gradients(self.X, self.Y, self.w, self.b)
            mw = self.beta1 * mw + (1 - self.beta1) * dw
            mb = self.beta1 * mb + (1 - self.beta1) * db
            vw = self.beta2 * vw + (1 - self.beta2) * dw**2
            vb = self.beta2 * vb + (1 - self.beta2) * db**2
            mw_hat = mw / (1 - self.beta1**t)
            mb_hat = mb / (1 - self.beta1**t)
            vw_hat = vw / (1 - self.beta2**t)
            vb_hat = vb / (1 - self.beta2**t)
            self.w -= self.lr * mw_hat / (np.sqrt(vw_hat) + self.epsilon)
            self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.epsilon)
            loss = compute_loss(self.X, self.Y, self.w, self.b)
            self.loss_history.append(loss)
            self.w_history.append(self.w)
            self.b_history.append(self.b)
        return self


# ==================== VISUALIZATION ====================

def plot_contour_with_variants(X, Y, variants_list):
    """Plot contour map with all gradient descent paths"""
    
    # Create weight and bias meshgrid for contour
    w_range = np.linspace(-4, 6, 100)
    b_range = np.linspace(-4, 6, 100)
    W, B = np.meshgrid(w_range, b_range)
    
    # Calculate loss for each point
    Z = np.zeros_like(W)
    for i in range(len(w_range)):
        for j in range(len(b_range)):
            Z[j, i] = compute_loss(X, Y, W[j, i], B[j, i])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 6))
    
    # Contour plot with all variants
    ax1 = fig.add_subplot(121)
    CS = ax1.contour(W, B, Z, levels=30, cmap='viridis')
    ax1.clabel(CS, inline=True, fontsize=8)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for idx, variant in enumerate(variants_list):
        w_hist = np.array(variant.w_history)
        b_hist = np.array(variant.b_history)
        ax1.plot(w_hist, b_hist, 'o-', linewidth=2, markersize=5, 
                label=variant.name, color=colors[idx], alpha=0.7)
        ax1.plot(w_hist[-1], b_hist[-1], 's', markersize=12, color=colors[idx])
    
    ax1.set_xlabel('Weight (w)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Bias (b)', fontsize=12, fontweight='bold')
    ax1.set_title('Gradient Descent Variants on Contour Map', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Loss curves comparison
    ax2 = fig.add_subplot(122)
    for idx, variant in enumerate(variants_list):
        ax2.plot(variant.loss_history, linewidth=2, label=variant.name, 
                color=colors[idx], alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss (MSE)', fontsize=12, fontweight='bold')
    ax2.set_title('Loss Convergence Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gradient_descent_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Contour plot saved as 'gradient_descent_comparison.png'")
    plt.show()


def print_results_table(variants_list):
    """Print comparison table of all variants"""
    results = []
    for variant in variants_list:
        metrics = variant.get_metrics()
        results.append(metrics)
    
    df = pd.DataFrame(results)
    print("\n" + "="*100)
    print("GRADIENT DESCENT VARIANTS - PERFORMANCE COMPARISON")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    print(f"\nBest R² Score: {df['r2_score'].idxmax()} - {df.loc[df['r2_score'].idxmax(), 'name']} ({df['r2_score'].max():.6f})")
    print(f"Lowest RMSE: {df['rmse'].idxmin()} - {df.loc[df['rmse'].idxmin(), 'name']} ({df['rmse'].min():.6f})")
    print(f"Lowest Final Loss: {df['final_loss'].idxmin()} - {df.loc[df['final_loss'].idxmin(), 'name']} ({df['final_loss'].min():.6f})")
    print("="*100 + "\n")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    print("\n" + "="*100)
    print("GRADIENT DESCENT VARIANTS - AUTOMATIC COMPARISON AND VISUALIZATION")
    print("="*100 + "\n")
    
    # Generate data
    print("📊 Generating synthetic dataset...")
    X, Y = generate_data(n_samples=100)
    print(f"✓ Dataset created: {X.shape[0]} samples, {X.shape[1]} feature(s)\n")
    
    # Initialize all variants with same hyperparameters
    learning_rate = 0.01
    epochs = 500
    
    print("🚀 Training all gradient descent variants...\n")
    variants = [
        BatchGradientDescent(X, Y, lr=learning_rate, epochs=epochs, name="Batch GD"),
        StochasticGradientDescent(X, Y, lr=learning_rate, epochs=epochs, name="Stochastic GD"),
        MiniBatchGradientDescent(X, Y, lr=learning_rate, epochs=epochs, batch_size=16, name="Mini-batch GD"),
        MomentumGradientDescent(X, Y, lr=learning_rate, epochs=epochs, momentum=0.9, name="Momentum"),
        RMSpropGradientDescent(X, Y, lr=learning_rate, epochs=epochs, name="RMSprop"),
        AdamGradientDescent(X, Y, lr=0.1, epochs=epochs, beta1=0.8, beta2=0.9, name="Adam")  # Adam with aggressive parameters
    ]
    
    # Train all variants
    for variant in variants:
        variant.train()
        print(f"✓ {variant.name} training completed")
    
    # Print results
    print_results_table(variants)
    
    # Plot results
    print("📈 Generating contour map with variant paths...")
    plot_contour_with_variants(X, Y, variants)
    
    print("\n" + "="*100)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*100 + "\n")