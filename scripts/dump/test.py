import cv2
import numpy as np
import matplotlib.pyplot as plt

def extend_line_segment(og_line, new_line):
    """
    Projects the new points onto the existing line without changing the line equation.
    Returns the updated line segment.
    """
    x1, y1, x2, y2 = og_line
    new_x1, new_y1, new_x2, new_y2 = new_line
    # Calculate the direction vector of the original line
    direction = np.array([x2 - x1, y2 - y1])
    direction = direction / np.linalg.norm(direction)  # Normalize the direction vector

    # Calculate the projections of the new points onto the direction vector
    proj1 = np.dot([new_x1 - x1, new_y1 - y1], direction)
    print(proj1)
    proj2 = np.dot([new_x2 - x2, new_y2 - y2], direction)
    print(proj2)
    print(np.linalg.norm([x2 - x1, y2 - y1]))



    # Update the line segment based on the projections
    if proj1 < 0:
        x1 = int(x1 + proj1 * direction[0])
        y1 = int(y1 + proj1 * direction[1])
    if proj2 > 0:
        x2 = int(x2 + proj2 * direction[0])
        y2 = int(y2 + proj2 * direction[1])

    return x1, y1, x2, y2

# Example usage
if __name__ == "__main__":
    # Create a dummy image
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Define some dummy lines
    dummy_lines = np.array([
        [100, 100, 200, 200],
        [90, 91, 185, 179]
    ])

    # Extend the second line to the first line
    merged = extend_line_segment(dummy_lines[0], dummy_lines[1])

    for line in dummy_lines:
        cv2.line(dummy_image, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3)
    cv2.line(dummy_image, (merged[0], merged[1]), (merged[2], merged[3]), (0, 255, 0), 3)
    plt.imshow(cv2.cvtColor(dummy_image, cv2.COLOR_BGR2RGB))
    plt.show()