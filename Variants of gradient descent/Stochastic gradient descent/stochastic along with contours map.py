import numpy as np
import matplotlib.pyplot as plt

class StochasticGradientDescent:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        """Train the model using SGD"""
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0
        
        for epoch in range(self.epochs):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Mini-batch updates
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                
                # Forward pass
                predictions = np.dot(X_batch, self.weights) + self.bias
                
                # Compute loss (MSE)
                loss = np.mean((predictions - y_batch) ** 2)
                epoch_loss += loss
                
                # Compute gradients
                dw = (2 / self.batch_size) * np.dot(X_batch.T, (predictions - y_batch))
                db = (2 / self.batch_size) * np.sum(predictions - y_batch)
                
                # Update weights and bias
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            self.loss_history.append(epoch_loss / (n_samples // self.batch_size))
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {self.loss_history[-1]:.6f}")
    
    def predict(self, X):
        """Make predictions"""
        return np.dot(X, self.weights) + self.bias


# Generate random training data
np.random.seed(42)
X_train = np.random.randn(100, 3)
true_weights = np.array([2.5, -1.3, 0.8])
y_train = np.dot(X_train, true_weights) + np.random.randn(100) * 0.1

# Create and train the model
print("Training Stochastic Gradient Descent Model...")
sgd = StochasticGradientDescent(learning_rate=0.01, epochs=100, batch_size=10)
sgd.fit(X_train, y_train)

# Make predictions on new random data
X_test = np.random.randn(10, 3)
y_pred = sgd.predict(X_test)

print("\n=== Results ===")
print(f"True weights: {true_weights}")
print(f"Learned weights: {sgd.weights}")
print(f"\nTest predictions:")
for i, pred in enumerate(y_pred):
    print(f"  Sample {i+1}: {pred:.4f}")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(sgd.loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Stochastic Gradient Descent Training Loss')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot contour map for 2D visualization (using first 2 features)
print("\n=== Generating Contour Map ===")
X_train_2d = X_train[:, :2]
true_weights_2d = true_weights[:2]
y_train_2d = np.dot(X_train_2d, true_weights_2d) + np.random.randn(100) * 0.1

# Train a 2D model for contour visualization
sgd_2d = StochasticGradientDescent(learning_rate=0.01, epochs=100, batch_size=10)
sgd_2d.fit(X_train_2d, y_train_2d)

# Create contour plot
w1_range = np.linspace(-2, 5, 100)
w2_range = np.linspace(-4, 2, 100)
w1_grid, w2_grid = np.meshgrid(w1_range, w2_range)
loss_grid = np.zeros_like(w1_grid)

# Calculate loss for each weight combination
for i in range(w1_grid.shape[0]):
    for j in range(w1_grid.shape[1]):
        w1, w2 = w1_grid[i, j], w2_grid[i, j]
        predictions = np.dot(X_train_2d, np.array([w1, w2]))
        loss_grid[i, j] = np.mean((predictions - y_train_2d) ** 2)

# Plot contours
plt.figure(figsize=(10, 8))
contour = plt.contour(w1_grid, w2_grid, loss_grid, levels=20, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)
plt.colorbar(contour, label='Loss (MSE)')
plt.xlabel('Weight 1')
plt.ylabel('Weight 2')
plt.title('SGD Loss Contour Map (2D Feature Space)')
plt.plot(sgd_2d.weights[0], sgd_2d.weights[1], 'r*', markersize=15, label='Learned Weights')
plt.plot(true_weights_2d[0], true_weights_2d[1], 'g^', markersize=10, label='True Weights')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"2D Model - True weights: {true_weights_2d}")
print(f"2D Model - Learned weights: {sgd_2d.weights}")
