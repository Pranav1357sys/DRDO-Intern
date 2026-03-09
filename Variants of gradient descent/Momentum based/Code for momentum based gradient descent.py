'''Code for momentum based gradient descent : '''
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

def do_momentum_gradient_descent(X, Y, init_w=0.0, init_b=0.0, learning_rate=1.0, max_epochs=1000, gamma=0.9):
    """
    Momentum-based gradient descent optimizer
    
    Args:
        X: Input features
        Y: Target values
        init_w: Initial weight
        init_b: Initial bias
        learning_rate: Learning rate (eta)
        max_epochs: Number of epochs
        gamma: Momentum coefficient (typically 0.9)
    
    Returns:
        w, b: Optimized parameters
        loss_history: Loss at each epoch
    """
    w, b = init_w, init_b
    eta = learning_rate
    prev_v_w, prev_v_b = 0, 0
    loss_history = []
    
    for i in range(max_epochs):
        # Compute gradients
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        
        # Momentum update
        v_w = gamma * prev_v_w + eta * dw
        v_b = gamma * prev_v_b + eta * db
        
        # Update parameters
        w = w - v_w
        b = b - v_b
        
        # Store for next iteration
        prev_v_w = v_w
        prev_v_b = v_b
        
        # Track loss
        loss = compute_loss(w, b, X, Y)
        loss_history.append(loss)
    
    return w, b, loss_history

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.array([1, 2, 3, 4, 5])
    Y = np.array([2, 4, 6, 8, 10])  # y = 2*x (approximately)
    
    # Train model
    w, b, losses = do_momentum_gradient_descent(
        X, Y, 
        init_w=0.0, 
        init_b=0.0, 
        learning_rate=1.0, 
        max_epochs=1000, 
        gamma=0.9
    )
    
    print(f"\nOptimized parameters: w={w:.4f}, b={b:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")

