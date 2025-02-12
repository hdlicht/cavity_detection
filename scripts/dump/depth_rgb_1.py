import subprocess
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import threading
import queue
import numpy as np
import pyelsed
from sklearn.cluster import DBSCAN


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

def dbscan_cluster(lines, rho_thresh=10, theta_thresh=5):
    """
    Cluster lines using DBSCAN based on rho and theta.

    Args:
    - lines: List of lines, where each line is [x1, y1, x2, y2].
    - rho_thresh: Threshold for rho difference.
    - theta_thresh: Threshold for angular (theta) difference in degrees.

    Returns:
    - labels: Cluster labels for the filtered lines.
    """
    # Extract endpoints
    x1, y1, x2, y2 = lines.T
    
    # Compute rho and theta for each line
    theta = np.arctan2(y2 - y1, x2 - x1)  # Slope angles in radians
    theta_deg = np.degrees(theta) % 180   # Map angles to [0, 180)
    rho = (x1 * np.cos(theta) + y1 * np.sin(theta))  # Compute rho for the line

    # Pairwise differences
    rho_diff = np.abs(rho[:, None] - rho)            # Pairwise rho difference
    theta_diff = np.abs(theta_deg[:, None] - theta_deg)  # Pairwise theta difference
    theta_diff = np.minimum(theta_diff, 180 - theta_diff)  # Handle angle wrapping

    # Build a distance matrix based on thresholds
    distance_matrix = (rho_diff / rho_thresh) + (theta_diff / theta_thresh)
    print(distance_matrix.shape)

    # Run DBSCAN with precomputed distances
    dbscan = DBSCAN(eps=1.0, min_samples=2, metric="precomputed")
    labels = dbscan.fit_predict(distance_matrix)

    return labels



def process_fusion(event):
    """Periodically process the fusion of RGB and Depth images."""
    with buffer_lock:
        if data_buffer["rgb"] is not None and data_buffer["depth"] is not None:
            rgb_image = data_buffer["rgb"]
            H = rgb_image.shape[0]
            W = rgb_image.shape[1]
            bilateral = cv2.bilateralFilter(rgb_image, d=7, sigmaColor=75, sigmaSpace=75)
            gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
            edges = cv2.Canny(bilateral, 20, 80)
            edges = cv2.dilate(edges, None, iterations=1)
            hough_image = np.zeros_like(rgb_image)
            # Detect lines using HoughLinesP
            image_lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=80, maxLineGap=20)
            print(image_lines.shape)
            # Draw lines on the image
            if image_lines is not None:
                image_lines = np.array(image_lines[:, 0])
                image_lines = image_lines[np.logical_and(image_lines[:, 3] > (2 * H // 3),image_lines[:, 1] > H//3)]
                    # cv2.line(hough_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            depth_image = data_buffer["depth"]
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_canny = cv2.Canny(depth_normalized, 20, 80)
            depth_canny = cv2.dilate(depth_canny, None, iterations=1)
            depth_lines = cv2.HoughLinesP(depth_canny, 1, np.pi/180, threshold=100, minLineLength=80, maxLineGap=20)

            # Draw lines on the image
            if depth_lines is not None:
                depth_lines = np.array(depth_lines[:, 0])
                depth_lines = depth_lines[np.logical_and(depth_lines[:, 3] > (2 * H // 3),depth_lines[:, 1] > H//3)]

            # Check for NaN, infinity, or too large values
            if np.any(np.isnan(depth_lines)) or np.any(np.isinf(depth_lines)) or np.any(np.abs(depth_lines) > np.finfo(np.float64).max):
                raise ValueError("depth_lines contains NaN, infinity, or a value too large for dtype('float64')")

            for x1, y1, x2, y2 in image_lines:
                # make one line from all lines in cluster
                cv2.line(hough_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            for x1, y1, x2, y2 in depth_lines:
                cv2.line(hough_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Apply DBSCAN with custom metric
            #dbscan = DBSCAN(eps=0.5, min_samples=2, metric=similar_lines).fit(line_features)

            all_lines = np.concatenate([image_lines, depth_lines], axis=0)

            if all_lines.shape[0] > 0:
                # Cluster labels for each line
                labels = dbscan_cluster(all_lines)

                # Group lines by clusters
                clusters = {}
                for idx, label in enumerate(labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(all_lines[idx])

                # Print the clustered lines
                for cluster_id, cluster_lines in clusters.items():
                    # make one line from all lines in cluster
                    cluster_lines = np.array(cluster_lines)
                    # get average slope for the cluster
                    x1, y1, x2, y2 = cluster_lines.T
                    epsilon = 1e-6
                    slope = np.mean((y2 - y1) / (x2 - x1 + epsilon))
                    if np.abs(slope) < 1e-6:
                        continue
                    y1 = np.min(y1)
                    y2 = np.max(y2)
                    middle_y = (y1 + y2) // 2
                    middle_x = (np.min(x1) + np.max(x2)) // 2
                    # draw line with average slope and top and bottom points through middle x
                    x1 = int(middle_x - (middle_y - y1) / slope)
                    x2 = int(middle_x + (y2 - middle_y) / slope)

                    cv2.line(rgb_image, (x1, y1), (x2, y2), (0, 0, 255), 3)

            cv2.imshow("image", rgb_image)
            cv2.imshow("Hough Lines", hough_image)
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