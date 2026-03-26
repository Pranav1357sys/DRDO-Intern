import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Set random seed for reproducibility
np.random.seed(42)

class MiniBatchGradientDescent:
    def __init__(self, learning_rate=0.01, batch_size=10, num_epochs=100):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.history = []
        self.weights_history = []
        
    def loss_function(self, X, y, weights):
        """Compute Mean Squared Error loss"""
        predictions = X.dot(weights)
        error = predictions - y
        loss = np.mean(error ** 2)
        return loss
    
    def gradient(self, X, y, weights):
        """Compute gradient of loss with respect to weights"""
        predictions = X.dot(weights)
        error = predictions - y
        gradient = 2 * X.T.dot(error) / len(X)
        return gradient
    
    def fit(self, X, y):
        """Train using mini-batch gradient descent"""
        num_samples = len(X)
        num_batches = max(1, num_samples // self.batch_size)
        
        # Initialize weights randomly
        self.weights = np.random.randn(X.shape[1]) * 0.01
        self.weights_history.append(self.weights.copy())
        
        for epoch in range(self.num_epochs):
            # Shuffle data
            indices = np.random.permutation(num_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch updates
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Compute gradient and update weights
                grad = self.gradient(X_batch, y_batch, self.weights)
                self.weights -= self.learning_rate * grad
                self.weights_history.append(self.weights.copy())
            
            # Record loss for each epoch
            loss = self.loss_function(X, y, self.weights)
            self.history.append(loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss:.6f}")
        
        print(f"Final Loss: {self.history[-1]:.6f}")
        return self
    
    def predict(self, X):
        return X.dot(self.weights)

# Generate random training data
np.random.seed(42)
X = np.random.randn(100, 2)
true_weights = np.array([3.5, -2.1])
y = X.dot(true_weights) + np.random.randn(100) * 0.5

# Train the model
print("Training Mini-Batch Gradient Descent...")
mbgd = MiniBatchGradientDescent(learning_rate=0.1, batch_size=10, num_epochs=100)
mbgd.fit(X, y)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Loss over epochs
ax1 = axes[0]
ax1.plot(mbgd.history, 'b-', linewidth=2, label='Training Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (MSE)', fontsize=12)
ax1.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Contour plot with optimization path
ax2 = axes[1]

# Create contour plot for 2D visualization
w0 = np.linspace(-2, 6, 100)
w1 = np.linspace(-5, 2, 100)
W0, W1 = np.meshgrid(w0, w1)
Z = np.zeros_like(W0)

for i in range(len(w0)):
    for j in range(len(w1)):
        weights_test = np.array([W0[j, i], W1[j, i]])
        Z[j, i] = mbgd.loss_function(X, y, weights_test)

# Plot contours
contour = ax2.contour(W0, W1, Z, levels=20, cmap='viridis', alpha=0.6)
ax2.clabel(contour, inline=True, fontsize=8)
contourf = ax2.contourf(W0, W1, Z, levels=20, cmap='viridis', alpha=0.3)

# Plot optimization path
weights_history = np.array(mbgd.weights_history)
ax2.plot(weights_history[:, 0], weights_history[:, 1], 'r.-', 
         linewidth=1, markersize=4, label='Optimization Path', alpha=0.7)

# Mark start and end points
ax2.plot(weights_history[0, 0], weights_history[0, 1], 'go', 
         markersize=12, label='Start Point', zorder=5)
ax2.plot(weights_history[-1, 0], weights_history[-1, 1], 'r*', 
         markersize=20, label='End Point', zorder=5)

# Mark true weights
ax2.plot(true_weights[0], true_weights[1], 'b^', 
         markersize=12, label='True Weights', zorder=5)

ax2.set_xlabel('Weight 0', fontsize=12)
ax2.set_ylabel('Weight 1', fontsize=12)
ax2.set_title('Contour Plot with Optimization Path', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Add colorbar
plt.colorbar(contourf, ax=ax2, label='Loss')

plt.tight_layout()
plt.savefig('mini_batch_gd.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFinal Weights:", mbgd.weights)
print("True Weights: ", true_weights)
print("Weight Error: ", np.abs(mbgd.weights - true_weights))
