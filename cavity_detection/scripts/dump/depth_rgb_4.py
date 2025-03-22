import subprocess
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import threading
import queue
import numpy as np
import pyelsed


# Global shared buffer
data_buffer = {"rgb": None, "depth": None}
buffer_lock = threading.Lock()

# Set the desired frequency for fusion (e.g., 10 Hz)
FUSION_RATE = 10  # 10 times per second (i.e., 100 ms period)

K = np.array([[570.342, 0.0,     314.5], 
              [0.0,     570.342, 235.5],
              [0.0,     0.0,     1.0]])

def get_depth(x, y):
    """Get the depth value at pixel (x, y) from the depth image."""
    with buffer_lock:
        depth_image = data_buffer["depth"]
        if depth_image is not None:
            return depth_image[y, x]
    return None

import numpy as np
import cv2

def normalize_lines(lines):
    # Ensure lines have consistent ordering: (x1, y1, x2, y2) where x1 < x2 or (x1 == x2 and y1 < y2)
    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
    swap = (x1 > x2) | ((x1 == x2) & (y1 > y2))
    normalized = lines.copy()
    normalized[swap] = lines[swap][:, [2, 3, 0, 1]]
    return normalized

def calculate_angles(lines):
    # Calculate angles of lines in degrees
    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
    angles = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return angles % 360  # Keep angles in [0, 360)

import numpy as np

def initialize_line_segment(x1, y1, x2, y2):
    """
    Initialize a line segment given endpoints (x1, y1) and (x2, y2).
    Outputs rho, theta, a, b for the infinite line and segment extent.
    """
    theta = np.arctan2(y2 - y1, x2 - x1)
    rho = x1 * np.cos(theta) + y1 * np.sin(theta)

    a = (x1 * np.cos(theta) + y1 * np.sin(theta)) - rho
    b = (x2 * np.cos(theta) + y2 * np.sin(theta)) - rho

    if a > b:
        a, b = b, a

    return rho, theta, a, b

def extend_segment(rho, theta, a, b, x, y):
    """
    Extend the segment (defined by rho, theta, a, b) to include a new point (x, y).
    """
    projection = (x * np.cos(theta) + y * np.sin(theta)) - rho
    a = min(a, projection)
    b = max(b, projection)
    return a, b

def merge_lines(lines, angle_threshold=5.0, distance_threshold=10.0):
    """
    Merge lines based on similarity in angle and proximity.
    """
    angle_threshold = np.radians(angle_threshold)  # Convert to radians
    merged_lines = []

    for line in lines:
        x1, y1, x2, y2 = line
        rho, theta, a, b = initialize_line_segment(x1, y1, x2, y2)

        matched = False
        for i, (mrho, mtheta, ma, mb) in enumerate(merged_lines):
            # Check angle similarity
            if abs(theta - mtheta) > angle_threshold:
                continue

            # Check if the line's projections overlap in space
            if abs(rho - mrho) > distance_threshold:
                continue

            # Extend the matched line
            ma, mb = extend_segment(mrho, mtheta, ma, mb, x1, y1)
            ma, mb = extend_segment(mrho, mtheta, ma, mb, x2, y2)
            merged_lines[i] = (mrho, mtheta, ma, mb)
            matched = True
            break

        # If no match, add this line as a new cluster
        if not matched:
            (x1, y1, x2, y2) = convert_to_cartesian(rho, theta, a, b)
            merged_lines.append((x1, y1, x2, y2))

    merged_lines = np.array(merged_lines, dtype=np.int32)
    
    return merged_lines

def convert_to_cartesian(rho, theta, a, b):
    """
    Convert a line segment defined by (rho, theta, a, b) back to cartesian endpoints (x1, y1) and (x2, y2).
    """
    # Unit vector along the line
    ux = np.cos(theta)
    uy = np.sin(theta)

    # Start and end points in Cartesian coordinates
    x1 = rho * ux + a * ux
    y1 = rho * uy + a * uy
    x2 = rho * ux + b * ux
    y2 = rho * uy + b * uy

    return (x1, y1, x2, y2)

def process_fusion(event):
    """Periodically process the fusion of RGB and Depth images."""
    with buffer_lock:
        if data_buffer["rgb"] is not None and data_buffer["depth"] is not None:
            rgb_image = data_buffer["rgb"]
            H, W = rgb_image.shape[:2]
            bilateral = cv2.bilateralFilter(rgb_image, d=9, sigmaColor=75, sigmaSpace=75)
            gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 100)
            edges = cv2.dilate(edges, None, iterations=1)
            hough_image = np.zeros_like(rgb_image)
            line_image = np.zeros_like(rgb_image)

            # Detect lines using HoughLinesP
            image_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=20)
            if image_lines is None:
                return
            image_lines = np.array(image_lines[:, 0])
            image_lines = image_lines[np.logical_and(image_lines[:, 3] > (2 * H // 3), image_lines[:, 1] > H // 3)]
            
            # Draw lines on the image
            merged_lines = merge_lines(image_lines, angle_threshold=10, distance_threshold=10)
            image_lines = image_lines.reshape(-1, 2, 2)
            merged_lines = merged_lines.reshape(-1, 2, 2)

            line_image = cv2.polylines(line_image, image_lines, isClosed=False, color=(0, 0, 255), thickness=2)
            hough_image = cv2.polylines(hough_image, merged_lines, isClosed=False, color=(0, 255, 0), thickness=2)

            cv2.imshow("Hough Lines", line_image)
            cv2.waitKey(1)

            cv2.imshow("Hough Rectangles", hough_image)
            cv2.waitKey(1)

            cv2.imshow("RGB Image", rgb_image)
            cv2.waitKey(1)
        else:
            rospy.logwarn("Waiting for both RGB and Depth images to be received.")

def rgb_callback(msg):
    """Callback function for RGB images."""
    try:
        rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with buffer_lock:
            data_buffer["rgb"] = rgb_image
    except Exception as e:
        rospy.logerr(f"Error in RGB callback: {e}")

def depth_callback(msg):
    """Callback function for Depth images."""
    try:
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with buffer_lock:
            data_buffer["depth"] = depth_image
    except Exception as e:
        rospy.logerr(f"Error in Depth callback: {e}")

if __name__ == "__main__":
    rospy.init_node("image_fusion_node")
    bridge = CvBridge()
    
    # Subscribe to RGB and Depth topics
    video_topic = "/camera/rgb/image_raw"
    depth_topic = "/camera/depth/image"
    rospy.Subscriber(video_topic, Image, rgb_callback, queue_size=10)
    rospy.Subscriber(depth_topic, Image, depth_callback, queue_size=10)

    # Timer to call process_fusion() periodically (e.g., every 100ms)
    rospy.Timer(rospy.Duration(1.0 / FUSION_RATE), process_fusion)

    rospy.spin()