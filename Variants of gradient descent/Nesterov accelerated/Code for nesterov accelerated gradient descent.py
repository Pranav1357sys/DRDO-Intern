'''Code for nesterov accelerated gradient descent : '''
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

def do_nesterov_accelerated_gradient_descent(X, Y, init_w=0.0, init_b=0.0, learning_rate=0.01, max_epochs=1000, gamma=0.9):
    """
    Nesterov-accelerated gradient descent optimizer
    
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
        dw, db = 0, 0
        #do partial updates
        v_w = gamma * prev_v_w
        v_b = gamma * prev_v_b
        # Compute gradients after partial updates
        for x, y in zip(X, Y):
            dw += grad_w(w-v_w, b-v_b, x, y)
            db += grad_b(w-v_w, b-v_b, x, y)
        
        # Average the gradients
        dw /= len(X)
        db /= len(X)
        
        # Nesterov update
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
    X = np.array([1, 3, 5, 7, 9])
    Y = np.array([2, 4, 6, 8, 10])  # y = 2*x (approximately)
    
    # Train model
    w, b, losses = do_nesterov_accelerated_gradient_descent(
        X, Y, 
        init_w=0.0, 
        init_b=0.0, 
        learning_rate=0.01,
        max_epochs=1000,
        gamma=0.9
    )
    
    print(f"\nOptimized parameters: w={w:.4f}, b={b:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")

