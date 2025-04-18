import numpy as np
import matplotlib.pyplot as plt

# Tank dimensions
L = 100  # Length of tank
W = 20   # Width of tank

# Gaussian parameters
sigma = 5                  # Spread of the spray
A = 1                      # Spray intensity per unit time
step_size = sigma * 1.5    # Distance to move for 50-70% overlap
num_steps = int(L // step_size)

# Offset between robot and spray center
offset = 10  # Distance between robot and spray center

# Create an empty tank
tank = np.zeros((L, W))

# Gaussian spray function
def spray_pattern(x, y, cx, cy, A, sigma):
    return A * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))

# Simulate robot moving and spraying
x, y = np.meshgrid(np.arange(L), np.arange(W), indexing='ij')
for i in range(num_steps):
    robot_pos = int(i * step_size)
    cx = robot_pos + offset  # Shift center by offset
    if cx < L:  # Avoid spraying beyond the tank
        tank += spray_pattern(x, y, cx, W // 2, A, sigma)

# Normalize for uniformity
tank /= np.max(tank)

# Plot the result
plt.imshow(tank.T, origin='lower', cmap='viridis', aspect='auto')
plt.colorbar(label='Fill Level')
plt.title('Tank Fill Simulation with Spray Offset')
plt.xlabel('Length')
plt.ylabel('Width')
plt.show()
