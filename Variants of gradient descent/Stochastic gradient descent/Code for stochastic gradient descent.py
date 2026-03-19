'''Code for stochastic gradient descent : '''
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

def do_stochastic_gradient_descent(X, Y, init_w=0.0, init_b=0.0, learning_rate=0.01, max_epochs=1000):
    """
    Stochastic gradient descent optimizer
    
    Args:
        X: Input features
        Y: Target values
        init_w: Initial weight
        init_b: Initial bias
        learning_rate: Learning rate (eta)
        max_epochs: Number of epochs
    
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
        
        for x, y in zip(X_shuffled, Y_shuffled):
            # Compute gradients for this sample
            dw = grad_w(w, b, x, y)
            db = grad_b(w, b, x, y)
            
            # Update parameters
            w = w - eta * dw
            b = b - eta * db
        
        # Track loss after each epoch
        loss = compute_loss(w, b, X, Y)
        loss_history.append(loss)
    
    return w, b, loss_history

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.array([11, 13, 15, 17, 19])
    Y = np.array([12, 14, 16, 18, 20])  # y = x + 1 (exactly)
    
    # Train model
    w, b, losses = do_stochastic_gradient_descent(
        X, Y, 
        init_w=0.0, 
        init_b=0.0, 
        learning_rate=0.01,
        max_epochs=1000
    )
    
    print(f"\nOptimized parameters: w={w:.4f}, b={b:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")