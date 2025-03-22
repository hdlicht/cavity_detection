#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import threading
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import threading
from std_msgs.msg import Float32MultiArray


# Global shared buffer
data_buffer = {"rgb": None, "depth": None}
buffer_lock = threading.Lock()

# Set the desired frequency for fusion (e.g., 10 Hz)
FUSION_RATE = 10  # 10 times per second (i.e., 100 ms period)

K_rgb = np.array([[570.342, 0.0,     314.5], 
                [0.0,     570.342, 235.5],
                [0.0,     0.0,     1.0]])

K_depth = np.array([[570.342, 0.0,     319.5],
                    [0.0,     570.342, 239.5],
                    [0.0,     0.0,     1.0]])

# Transformation matrix from depth to RGB
T_depth_rgb = np.array([
    [1, 0, 0, -0.025], 
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1] ])

T_camera_world = np.array([
    [1,  0,  0,  0],  
    [0,  0, -1,  0],  
    [0,  1,  0,  0.6069], 
    [0,  0,  0,  1] ])

def get_depth(depth_image, x, y):
    """Get the depth value at pixel (x, y) from the depth image."""
    return depth_image[y, x]

def line_intersection(p1, p2, p3, p4):

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Compute the determinants
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator == 0:
        return None  # Lines are parallel, no intersection

    # Use Cramer's rule to find the intersection point
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    return (px, py)

def get_3d_point(depth_image, x, y):
    """Get the 3D point at pixel (x, y) using the depth image."""
    z = get_depth(depth_image, x, y)
    if z is None:
        return None
    x = (x - K_rgb[0, 2]) / K_rgb[0, 0] * z
    y = (y - K_rgb[1, 2]) / K_rgb[1, 1] * z

    # Apply camera transformation
    point = np.array([x, y, z, 1])
    point = T_camera_world @ point
    x, y, z = point[:3]

    return np.array([x, y, z])

def get_3d_line(depth_image, x1, y1, x2, y2):
    """Get the 3D line segment given endpoints (x1, y1) and (x2, y2)."""
    x1, y1, z1 = get_3d_point(depth_image, x1, y1)
    x2, y2, z2 = get_3d_point(depth_image, x2, y2)
    print(np.linalg.norm(np.array([x1, y1, z1]) - np.array([x2, y2, z2])))
    return np.array([x1, y1, z1, x2, y2, z2])


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
    proj2 = np.dot([new_x2 - x2, new_y2 - y2], direction)

    # Update the line segment based on the projections
    if proj1 < 0:
        x1 = int(x1 + proj1 * direction[0])
        y1 = int(y1 + proj1 * direction[1])
    if proj2 > 0:
        x2 = int(x2 + proj2 * direction[0])
        y2 = int(y2 + proj2 * direction[1])

    return x1, y1, x2, y2

def merge_lines(lines, angle_threshold=20, rectangle_width=20):
    if lines is None or len(lines) == 0:
        return None

    # Normalize lines
    normalized_lines = normalize_lines(lines)

    # Calculate lengths and angles
    lengths = np.sqrt((normalized_lines[:, 2] - normalized_lines[:, 0])**2 + 
                      (normalized_lines[:, 3] - normalized_lines[:, 1])**2)
    angles = calculate_angles(normalized_lines)

    # Sort lines by length (descending)
    sorted_indices = np.argsort(-lengths)
    normalized_lines = normalized_lines[sorted_indices]
    angles = angles[sorted_indices]

    used_lines = np.zeros(len(normalized_lines), dtype=bool)
    rectangles = []
    merged_lines = []
    counts = []

    for i, base_line in enumerate(normalized_lines):
        count = 0
        if used_lines[i]:
            continue
        if lengths[i] < 30:
            break

        base_angle = angles[i]

        # Create rectangle around the base line
        x1, y1, x2, y2 = base_line
        dx = rectangle_width * np.cos(np.radians(base_angle + 90))
        dy = rectangle_width * np.sin(np.radians(base_angle + 90))
        rect_points = np.array([
            [x1 + dx, y1 + dy],
            [x1 - dx, y1 - dy],
            [x2 - dx, y2 - dy],
            [x2 + dx, y2 + dy]
        ], dtype=np.int32)
        

        # Check for lines with similar angle and overlapping
        angle_diff = np.abs((angles - base_angle + 180) % 360 - 180)  # Angle difference in [0, 180]
        is_similar_angle = angle_diff <= angle_threshold

        for j, compare_line in enumerate(normalized_lines):
            if used_lines[j] or not is_similar_angle[j]:
                continue

            # Check overlap by comparing end points of the compare_line with the rectangle
            cx1, cy1, cx2, cy2 = compare_line
            # calculate the center of the compare line
            cx = (cx1 + cx2) / 2
            cy = (cy1 + cy2) / 2

            for pt in [(cx1, cy1), (cx2, cy2), (cx, cy)]:
                pt = np.array(pt, dtype=np.float32)
                if cv2.pointPolygonTest(rect_points, pt, False) >= 0:
                    used_lines[j] = True
                    count += 1
                    x1, y1, x2, y2 = extend_line_segment([x1, y1, x2, y2], compare_line)
                    rect_points = np.array([
                        [x1 + dx, y1 + dy],
                        [x1 - dx, y1 - dy],
                        [x2 - dx, y2 - dy],
                        [x2 + dx, y2 + dy]], dtype=np.int32)
                    break

        used_lines[i] = True
        merged_lines.append([x1, y1, x2, y2])
        rectangles.append(rect_points)
        counts.append(count)

    merged_lines = np.array(merged_lines)
    return merged_lines

def process_fusion(event):
    """Periodically process the fusion of RGB and Depth images."""
    with buffer_lock:
        if data_buffer["rgb"] is not None and data_buffer["depth"] is not None:
            rgb_image = data_buffer["rgb"]
            H, W = rgb_image.shape[:2]
            bilateral = cv2.bilateralFilter(rgb_image, d=9, sigmaColor=75, sigmaSpace=75)
            gaussian = cv2.GaussianBlur(rgb_image, (7, 7), 0)
            gray = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY)
            rgb_edges = cv2.Canny(gray, 30, 60)
            for i in range(1):
                rgb_edges = cv2.dilate(rgb_edges, None, iterations=1)
                rgb_edges = cv2.erode(rgb_edges, None, iterations=1)

            # Detect lines using HoughLinesP
            rgb_lines = cv2.HoughLinesP(rgb_edges, 1, np.pi / 180, threshold=10, minLineLength=30, maxLineGap=10)
            
            if rgb_lines is None:
                return
            rgb_lines = np.array(rgb_lines[:, 0])
            rgb_lines = rgb_lines[
                (np.minimum(rgb_lines[:, 1], rgb_lines[:, 3]) > H // 3) &  
                (np.maximum(rgb_lines[:, 1], rgb_lines[:, 3]) > 2 * H // 3)  
            ]    
            rgb_line_image = cv2.cvtColor(rgb_edges, cv2.COLOR_GRAY2BGR)
            for x1, y1, x2, y2 in rgb_lines:
                cv2.line(rgb_line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("RGB Image", rgb_image)
            cv2.imshow("Filtered", bilateral)
            cv2.imshow("Gauss", gaussian)
            cv2.imshow("RGB Lines", rgb_line_image)
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
    rospy.init_node("detect_lines_node")
    bridge = CvBridge()
    
    # Subscribe to RGB and Depth topics
    video_topic = "/camera/rgb/image_raw"
    depth_topic = "/camera/depth/image"
    rospy.Subscriber(video_topic, Image, rgb_callback, queue_size=10)
    rospy.Subscriber(depth_topic, Image, depth_callback, queue_size=10)
    pub = rospy.Publisher('/detected_lines', Float32MultiArray, queue_size=10)


    # Timer to call process_fusion() periodically (e.g., every 100ms)
    rospy.Timer(rospy.Duration(1.0 / FUSION_RATE), process_fusion)

    rospy.spin()