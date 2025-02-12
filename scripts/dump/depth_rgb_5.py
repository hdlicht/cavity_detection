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
            gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
            rgb_edges = cv2.Canny(gray, 40, 80)
            for i in range(1):
                rgb_edges = cv2.dilate(rgb_edges, None, iterations=1)
                rgb_edges = cv2.erode(rgb_edges, None, iterations=1)

            depth_image = data_buffer["depth"]
            too_far = np.logical_or(depth_image > 4.0, np.isnan(depth_image)) # Filter out values too far or NaN
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_edges = cv2.Canny(depth_normalized, 20, 50)
            depth_edges[too_far] = 0
            rgb_edges[too_far] = 0
            depth_edges = cv2.dilate(depth_edges, None, iterations=2)
            depth_edges = cv2.erode(depth_edges, None, iterations=2)

            all_edges = np.maximum(rgb_edges, depth_edges)
            all_lines = cv2.HoughLinesP(all_edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)
            if all_lines is None:
                return
            all_lines = all_lines.reshape(-1,4)
            all_lines = all_lines[
                (np.minimum(all_lines[:, 1], all_lines[:, 3]) > H // 3) &  
                (np.maximum(all_lines[:, 1], all_lines[:, 3]) > H // 2)  
            ]    


            # Detect lines using HoughLinesP
            rgb_lines = cv2.HoughLinesP(rgb_edges, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)
            depth_lines = cv2.HoughLinesP(depth_edges, 1, np.pi / 180, threshold=150, minLineLength=20, maxLineGap=10)\
            
            if rgb_lines is None:
                return
            rgb_lines = np.array(rgb_lines[:, 0])
            rgb_lines = rgb_lines[
                (np.minimum(rgb_lines[:, 1], rgb_lines[:, 3]) > H // 3) &  
                (np.maximum(rgb_lines[:, 1], rgb_lines[:, 3]) > 2 * H // 3)  
            ]    
            if depth_lines is None:
                return
            depth_lines = np.array(depth_lines[:, 0])
            depth_lines = depth_lines[
                (np.minimum(depth_lines[:, 1], depth_lines[:, 3]) > H // 3) & 
                (np.maximum(depth_lines[:, 1], depth_lines[:, 3]) > 2 * H // 3)  
            ]                
            # Draw lines on the image
            merged_lines = merge_lines(all_lines, angle_threshold=15, rectangle_width=8)
            if merged_lines is None or len(merged_lines) == 0:
                return
            merged_lines = np.array(merged_lines)
            lengths = np.sqrt((merged_lines[:, 2] - merged_lines[:, 0])**2 + (merged_lines[:, 3] - merged_lines[:, 1])**2)
            angles = np.degrees(np.arctan2(merged_lines[:, 3] - merged_lines[:, 1], merged_lines[:, 2] - merged_lines[:, 0]))
            # calculate a line perpendicular to the average angle of the 3 longest lines
            longest_lines = np.argsort(-lengths)[:3]
            avg_angle = np.mean(angles[longest_lines])
            mid_points = (merged_lines[:, :2] + merged_lines[:, 2:]) / 2
            xm, ym = np.mean(mid_points[longest_lines], axis=0)
            perp_angle = avg_angle + 90
            px1 = xm - 1000 * np.cos(np.radians(perp_angle))
            py1 = ym - 1000 * np.sin(np.radians(perp_angle))
            px2 = xm + 1000 * np.cos(np.radians(perp_angle))
            py2 = ym + 1000 * np.sin(np.radians(perp_angle))

            # Find all intersections with the perpendicular line
            intersections = []
            for line in merged_lines:
                x1, y1, x2, y2 = line
                intersection = line_intersection((x1, y1), (x2, y2), (px1, py1), (px2, py2))
                if intersection is not None:
                    intersections.append(intersection)

            merged_lines_3d = [get_3d_line(depth_image, x1, y1, x2, y2) for x1, y1, x2, y2 in merged_lines]
            merged_lines_3d = np.array(merged_lines_3d)

            pub = rospy.Publisher('/detected_3d_lines', Float32MultiArray, queue_size=10)
            msg = Float32MultiArray()
            msg.data = merged_lines_3d.flatten()
            pub.publish(msg)

            merged_lines = merged_lines.reshape(-1, 2, 2)
            flat = (np.abs(merged_lines_3d[:,2]- merged_lines_3d[:,5]) < 0.2)
            flat_lines = merged_lines[flat]
            not_flat_lines = merged_lines[~flat]
            merged_line_image = np.zeros_like(rgb_image)
            merged_line_image = cv2.polylines(merged_line_image, flat_lines, isClosed=False, color=(0, 0, 255), thickness=2)
            merged_line_image = cv2.polylines(merged_line_image, not_flat_lines, isClosed=False, color=(0, 255, 0), thickness=2)
            # merged_line_image = cv2.line(merged_line_image, (int(x1), int(y1)), (int(x2), int(y2),), (255, 0, 0), 2)
            # for x, y in intersections:
            #     cv2.circle(merged_line_image, (int(x), int(y)), 5, (0, 255, 255), -1)

            # paint rgb hough lines on top of rgb edges
            rgb_line_image = cv2.cvtColor(rgb_edges, cv2.COLOR_GRAY2BGR)
            rgb_lines = rgb_lines.reshape(-1, 2, 2)
            rgb_line_image = cv2.polylines(rgb_line_image, rgb_lines, isClosed=False, color=(0, 0, 255), thickness=2)

            # paint depth hough lines on top of depth edges
            depth_line_image = cv2.cvtColor(depth_edges, cv2.COLOR_GRAY2BGR)
            depth_lines = depth_lines.reshape(-1, 2, 2)
            depth_line_image = cv2.polylines(depth_line_image, depth_lines, isClosed=False, color=(0, 0, 255), thickness=2)

            # make one image with depth_edges and rgb_edges on different channels
            edges_image = np.zeros((H, W, 3), dtype=np.uint8)
            edges_image[..., 0] = depth_edges
            edges_image[..., 1] = rgb_edges

            cv2.imshow("RGB Lines", rgb_line_image)
            cv2.imshow("Depth lines", depth_line_image)
            cv2.imshow("Edges", edges_image)
            cv2.imshow("Merged Lines", merged_line_image)
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