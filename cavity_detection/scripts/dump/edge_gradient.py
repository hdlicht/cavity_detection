import cv2
import numpy as np

# Step 1: Load Image and Preprocess
edges = cv2.imread("Depth lines_screenshot_17.12.2024.png", cv2.IMREAD_GRAYSCALE)

# Invert the image so black becomes the foreground
#binary_image = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY_INV)[1]

# Apply connected components analysis
num_labels, labels = cv2.connectedComponents(edges, connectivity=8)

# Create output visualization
output_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
for label in range(1, num_labels):  # Skip background label 0
    mask = (labels == label).astype("uint8") * 255
    color = np.random.randint(0, 255, size=3).tolist()
    output_image[mask == 255] = color  # Random color for each connected region

# Show results
cv2.imshow("Original Image", edges)
cv2.imshow("Connected Components", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

