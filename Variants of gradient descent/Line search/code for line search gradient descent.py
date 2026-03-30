import numpy as np
import matplotlib.pyplot as plt

def objective_function(x):
    """Simple quadratic function: f(x) = (x-3)^2 + (x-2)^2"""
    return (x[0] - 3)**2 + (x[1] - 2)**2

def gradient(x):
    """Gradient of the objective function"""
    return np.array([2*(x[0] - 3), 2*(x[1] - 2)])

def line_search_gradient_descent(x0, learning_rate=0.1, max_iterations=100, tolerance=1e-6):
    """
    Line search gradient descent optimizer
    
    Args:
        x0: Initial point
        learning_rate: Initial learning rate
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
    
    Returns:
        x_history: History of iterations
        loss_history: History of loss values
    """
    x = x0.copy()
    x_history = [x.copy()]
    loss_history = [objective_function(x)]
    
    for iteration in range(max_iterations):
        # Compute gradient
        grad = gradient(x)
        grad_norm = np.linalg.norm(grad)
        
        if grad_norm < tolerance:
            print(f"Converged at iteration {iteration}")
            break
        
        # Line search: find step size
        step_size = learning_rate
        best_loss = objective_function(x)
        best_step_size = step_size
        
        # Simple line search: try different step sizes
        for _ in range(5):
            x_new = x - step_size * grad
            new_loss = objective_function(x_new)
            
            if new_loss < best_loss:
                best_loss = new_loss
                best_step_size = step_size
                step_size *= 1.2
            else:
                step_size *= 0.5
        
        # Update position with best step size
        x = x - best_step_size * grad
        x_history.append(x.copy())
        loss_history.append(objective_function(x))
        
        print(f"Iteration {iteration}: x = {x}, loss = {loss_history[-1]:.6f}")
    
    return np.array(x_history), np.array(loss_history)

# Generate random initial point
np.random.seed(42)
x0 = np.random.randn(2) * 5  # Random starting point

print(f"Starting point: {x0}")
print(f"Initial loss: {objective_function(x0):.6f}\n")

# Run line search gradient descent
x_history, loss_history = line_search_gradient_descent(x0, learning_rate=0.1, max_iterations=100)

print(f"\nFinal point: {x_history[-1]}")
print(f"Final loss: {loss_history[-1]:.6f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Loss over iterations
axes[0].plot(loss_history, 'b-o', linewidth=2, markersize=4)
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Loss')
axes[0].set_title('Loss over Iterations')
axes[0].grid(True)
axes[0].set_yscale('log')

# Plot 2: 2D contour plot with optimization path
x_range = np.linspace(-5, 8, 100)
y_range = np.linspace(-5, 8, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = (X - 3)**2 + (Y - 2)**2

axes[1].contour(X, Y, Z, levels=20, cmap='viridis')
axes[1].plot(x_history[:, 0], x_history[:, 1], 'r-o', linewidth=2, markersize=6, label='Optimization path')
axes[1].plot(x_history[0, 0], x_history[0, 1], 'go', markersize=10, label='Start')
axes[1].plot(x_history[-1, 0], x_history[-1, 1], 'r*', markersize=15, label='End')
axes[1].plot(3, 2, 'b*', markersize=15, label='Optimum')
axes[1].set_xlabel('x1')
axes[1].set_ylabel('x2')
axes[1].set_title('Optimization Path on Contour Plot')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
