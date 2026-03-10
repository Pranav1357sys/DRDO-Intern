import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Define a simple quadratic function for demonstration
def objective_function(x, y):
    """Quadratic function: f(x,y) = (x-2)^2 + (y+1)^2"""
    return (x - 2)**2 + (y + 1)**2

def gradient(x, y):
    """Gradient of the objective function"""
    df_dx = 2 * (x - 2)
    df_dy = 2 * (y + 1)
    return np.array([df_dx, df_dy])

def momentum_gradient_descent(start_point, learning_rate=0.01, momentum=0.9, iterations=100):
    """
    Momentum-based Gradient Descent
    
    Args:
        start_point: Initial point [x, y]
        learning_rate: Step size for updates
        momentum: Momentum coefficient (0 to 1)
        iterations: Number of iterations
    
    Returns:
        path: List of points visited during optimization
        losses: List of loss values at each iteration
    """
    path = [start_point.copy()]
    losses = [objective_function(start_point[0], start_point[1])]
    
    velocity = np.zeros_like(start_point)
    current_point = start_point.copy()
    
    for i in range(iterations):
        # Compute gradient at current point
        grad = gradient(current_point[0], current_point[1])
        
        # Update velocity with momentum
        velocity = momentum * velocity - learning_rate * grad
        
        # Update position
        current_point = current_point + velocity
        
        # Store path and loss
        path.append(current_point.copy())
        loss = objective_function(current_point[0], current_point[1])
        losses.append(loss)
    
    return np.array(path), np.array(losses)

# Set random seed for reproducibility
np.random.seed(42)

# Generate random starting point
start_x = np.random.uniform(-5, 5)
start_y = np.random.uniform(-5, 5)
start_point = np.array([start_x, start_y])

print(f"Starting point: ({start_x:.4f}, {start_y:.4f})")
print(f"Starting loss: {objective_function(start_x, start_y):.4f}")

# Run momentum gradient descent
path, losses = momentum_gradient_descent(
    start_point, 
    learning_rate=0.05, 
    momentum=0.9, 
    iterations=100
)

final_point = path[-1]
print(f"Final point: ({final_point[0]:.4f}, {final_point[1]:.4f})")
print(f"Final loss: {losses[-1]:.4f}")
print(f"Optimal point: (2.0000, -1.0000)")

# Create contour plot
fig, ax = plt.subplots(figsize=(10, 8))

# Generate contour data
x_range = np.linspace(-6, 8, 300)
y_range = np.linspace(-6, 4, 300)
X, Y = np.meshgrid(x_range, y_range)
Z = (X - 2)**2 + (Y + 1)**2

# Plot contours
contours = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
ax.clabel(contours, inline=True, fontsize=8)
contourf = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.3)

# Plot optimization path
ax.plot(path[:, 0], path[:, 1], 'r.-', linewidth=2, markersize=4, label='Optimization Path', alpha=0.7)

# Plot start and end points
ax.plot(path[0, 0], path[0, 1], 'go', markersize=12, label='Start Point', zorder=5)
ax.plot(path[-1, 0], path[-1, 1], 'r*', markersize=20, label='End Point', zorder=5)

# Plot optimal point
ax.plot(2, -1, 'b^', markersize=12, label='Optimal Point (2, -1)', zorder=5)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Momentum-based Gradient Descent on Quadratic Function', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(contourf, ax=ax)
cbar.set_label('Loss Value', fontsize=11)

plt.tight_layout()
plt.show()

# Plot loss curve
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(losses, 'b-', linewidth=2)
ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Loss Curve During Momentum-based Gradient Descent', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
