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

def dbscan_cluster(lines, slope_thresh=10, dist_thresh=10):
    """
    Cluster lines in the bottom half of an image using DBSCAN.
    
    Args:
    - lines: List of lines, where each line is [x1, y1, x2, y2].
    - y_thresh: Vertical threshold to filter lines in the bottom half.
    - slope_thresh: Threshold for angle differences (degrees).
    - dist_thresh: Threshold for midpoint distances.

    Returns:
    - labels: Cluster labels for the filtered lines.
    """
    # Filter lines in the bottom half

    # Extract endpoints
    x1, y1, x2, y2 = lines.T

    # Compute features: midpoints and slopes
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    slopes = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Slope in degrees

    # Pairwise feature comparison
    mid_dist = np.sqrt((mid_x[:, None] - mid_x)**2 + (mid_y[:, None] - mid_y)**2)
    slope_diff = np.abs(slopes[:, None] - slopes)
    slope_diff = np.minimum(slope_diff, 360 - slope_diff)

    # Build distance matrix based on thresholds
    distance_matrix = (slope_diff > slope_thresh) | (mid_dist > dist_thresh)
    distance_matrix = distance_matrix.astype(float)

    # Run DBSCAN with precomputed distances
    dbscan = DBSCAN(eps=0.5, min_samples=2, metric="precomputed")
    labels = dbscan.fit_predict(distance_matrix)

    return labels


def process_fusion(event):
    """Periodically process the fusion of RGB and Depth images."""

    with buffer_lock:
        if data_buffer["rgb"] is not None and data_buffer["depth"] is not None:
            rgb_image = data_buffer["rgb"]
            H = rgb_image.shape[0]
            W = rgb_image.shape[1]
            hough_image = np.zeros_like(rgb_image)

            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            segments, scores = pyelsed.detect(gray, sigma=1, gradientThreshold=5, minLineLen=50, lineFitErrThreshold=1)
            image_lines = np.array(segments).astype(np.int32)
            # image_lines = image_lines[np.logical_and(image_lines[:, 3] > (2 * H // 3),image_lines[:, 1] > H//3)]

            depth_image = data_buffer["depth"]
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            segments, scores = pyelsed.detect(depth_normalized, sigma=1, gradientThreshold=5, minLineLen=50, lineFitErrThreshold=1)
            depth_lines = np.array(segments).astype(np.int32)
            # depth_lines = depth_lines[np.logical_and(depth_lines[:, 3] > (2 * H // 3),depth_lines[:, 1] > H//3)]

            for x1, y1, x2, y2 in image_lines:
                # make one line from all lines in cluster
                cv2.line(hough_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            for x1, y1, x2, y2 in depth_lines:
                # make one line from all lines in cluster
                cv2.line(hough_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Apply DBSCAN with custom metric
            #dbscan = DBSCAN(eps=0.5, min_samples=2, metric=similar_lines).fit(line_features)

            all_lines = np.concatenate([image_lines, depth_lines], axis=0)

            # Cluster labels for each line
            labels = dbscan_cluster(all_lines)

            # Group lines by clusters
            clusters = {}
            for idx, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(all_lines[idx])

            for cluster_id, cluster_lines in clusters.items():
                cluster_lines = np.array(cluster_lines)
                x1, y1, x2, y2 = cluster_lines.T
                
                # Avoid division by zero when computing slopes
                epsilon = 1e-6
                slopes = (y2 - y1) / (x2 - x1 + epsilon)
                valid_mask = np.abs(slopes) < 1e6  # Filter out extreme slopes (e.g., near-vertical lines)

                if np.sum(valid_mask) == 0:  # If no valid slopes, skip
                    continue

                # Average slope and midpoints
                avg_slope = np.mean(slopes[valid_mask])
                
                if np.isnan(avg_slope) or np.isinf(avg_slope):  # Skip invalid clusters
                    continue

                # Compute the middle point of the cluster
                middle_x = (np.min(x1[valid_mask]) + np.max(x2[valid_mask])) // 2
                middle_y = (np.min(y1[valid_mask]) + np.max(y2[valid_mask])) // 2

                # Compute the corresponding y-values for the min and max y-coordinates
                min_y = np.min(y1[valid_mask])
                max_y = np.max(y2[valid_mask])

                # Calculate the corresponding x values based on the average slope
                x_min = middle_x - (middle_y - min_y) / avg_slope
                x_max = middle_x + (max_y - middle_y) / avg_slope

                if np.isnan(x_min) or np.isinf(x_max):  # Skip invalid clusters
                    continue
                # Draw the averaged line through the middle point with min/max y-values
                cv2.line(rgb_image, (int(x_min), min_y), (int(x_max), max_y), (0, 0, 255), 3)

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