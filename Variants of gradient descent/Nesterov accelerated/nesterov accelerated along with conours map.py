import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Nesterov Accelerated Gradient Descent Implementation
def nesterov_agd(f, grad_f, x0, learning_rate=0.01, momentum=0.9, num_iterations=100):
    """
    Nesterov Accelerated Gradient Descent optimizer
    
    Parameters:
    - f: objective function
    - grad_f: gradient function
    - x0: initial point
    - learning_rate: step size
    - momentum: momentum parameter (typically 0.9)
    - num_iterations: number of iterations
    
    Returns:
    - x: final point
    - history: list of all points visited
    """
    x = x0.copy()
    v = np.zeros_like(x)
    history = [x.copy()]
    
    for i in range(num_iterations):
        # Nesterov update: compute gradient at lookahead position
        x_lookahead = x - momentum * v
        grad = grad_f(x_lookahead)
        
        # Update velocity
        v = momentum * v + learning_rate * grad
        
        # Update position
        x = x - v
        
        history.append(x.copy())
    
    return x, np.array(history)

# Define objective function (Rosenbrock function)
def rosenbrock(x):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Gradient of Rosenbrock function
def grad_rosenbrock(x):
    """Gradient of Rosenbrock function"""
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return np.array([dx, dy])

# Random initial point
np.random.seed(42)
x0 = np.random.uniform(-2, 2, size=2)

print(f"Initial point: {x0}")
print(f"Initial function value: {rosenbrock(x0):.4f}")

# Run Nesterov AGD
final_point, trajectory = nesterov_agd(
    rosenbrock, 
    grad_rosenbrock, 
    x0, 
    learning_rate=0.001, 
    momentum=0.9, 
    num_iterations=500
)

print(f"Final point: {final_point}")
print(f"Final function value: {rosenbrock(final_point):.4f}")

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Create contour plot
x_range = np.linspace(-2, 2, 400)
y_range = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = (1 - X)**2 + 100 * (Y - X**2)**2

# Plot contours
contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
ax.clabel(contour, inline=True, fontsize=8)
contourf = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.3)

# Plot trajectory
ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, label='NAG Trajectory')
ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start Point')
ax.plot(trajectory[-1, 0], trajectory[-1, 1], 'r*', markersize=15, label='Final Point')
ax.plot(1, 1, 'b*', markersize=15, label='Optimal Point (1, 1)')

# Labels and formatting
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Nesterov Accelerated Gradient Descent on Rosenbrock Function', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.colorbar(contourf, ax=ax, label='Function Value')
plt.tight_layout()
plt.show()

# Plot convergence curve
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
function_values = [rosenbrock(point) for point in trajectory]
ax.semilogy(function_values, 'b-', linewidth=2)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Function Value (log scale)', fontsize=12)
ax.set_title('Convergence of Nesterov Accelerated Gradient Descent', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
