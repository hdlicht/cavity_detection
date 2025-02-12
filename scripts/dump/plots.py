import matplotlib.pyplot as plt
import numpy as np

# Data
lines = np.array(
[[1.42, 1.53, 2.26, 0.  ],
 [1.75, 1.69, 2.54, 0.26],
 [2.01, 1.86, 2.84, 0.23],
 [2.3,  2.08, 3.02, 0.51],
 [2.55, 2.2,  3.2,  0.86]]
)

# Plotting
plt.figure()
for line in lines:
    y1, x1, y2, x2 = line
    plt.plot([-x1, -x2], [y1, y2], marker='o')

plt.xlabel('y-axis')
plt.ylabel('x-axis')
plt.title('2D Line Plot')
plt.grid(True)
plt.show()