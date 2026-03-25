'''Code for mini-batch gradient descent : '''
import numpy as np

# Helper functions
def grad_w(w, b, x, y):
    """Gradient of loss with respect to w"""
    return -2 * x * (y - (w * x + b))

def grad_b(w, b, x, y):
    """Gradient of loss with respect to b"""
    return -2 * (y - (w * x + b))

def compute_loss(w, b, X, Y):
    """Compute mean squared error loss"""
    predictions = w * X + b
    return np.mean((Y - predictions) ** 2)

def do_mini_batch_gradient_descent(X, Y, init_w=0.0, init_b=0.0, learning_rate=0.01, max_epochs=1000, batch_size=5):
    """
    Mini-batch gradient descent optimizer
    
    Args:
        X: Input features
        Y: Target values
        init_w: Initial weight
        init_b: Initial bias
        learning_rate: Learning rate (eta)
        max_epochs: Number of epochs
        batch_size: Size of each mini-batch
    
    Returns:
        w, b: Optimized parameters
        loss_history: Loss at each epoch
    """
    w, b = init_w, init_b
    eta = learning_rate
    loss_history = []
    
    for i in range(max_epochs):
        # Shuffle the data for each epoch
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]
        
        for j in range(0, len(X), batch_size):
            # Get mini-batch
            X_batch = X_shuffled[j:j+batch_size]
            Y_batch = Y_shuffled[j:j+batch_size]
            
            dw, db = 0, 0
            # Compute gradients for the batch
            for x, y in zip(X_batch, Y_batch):
                dw += grad_w(w, b, x, y)
                db += grad_b(w, b, x, y)
            
            # Average the gradients over the batch
            dw /= len(X_batch)
            db /= len(Y_batch)
            
            # Update parameters
            w = w - eta * dw
            b = b - eta * db
        
        # Track loss after each epoch
        loss = compute_loss(w, b, X, Y)
        loss_history.append(loss)
    
    return w, b, loss_history

# Example usage with random inputs
if __name__ == "__main__":
    # Generate sample data with noise to showcase convergence
    np.random.seed(42)
    X = np.linspace(0, 10, 20)
    Y = 2 * X + 1 + np.random.normal(0, 0.5, 20)  # True: w=2, b=1, with Gaussian noise (std=0.5)
    
    # Train model
    w, b, losses = do_mini_batch_gradient_descent(
        X, Y, 
        init_w=0.0, 
        init_b=0.0, 
        learning_rate=0.01,
        max_epochs=2000,
        batch_size=5
    )
    
    print(f"\nOptimized parameters: w={w:.4f}, b={b:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"True parameters: w=2.0000, b=1.0000")