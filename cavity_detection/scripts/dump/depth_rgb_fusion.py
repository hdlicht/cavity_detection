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


def process_fusion(event):
    """Periodically process the fusion of RGB and Depth images."""
    with buffer_lock:

        if data_buffer["rgb"] is not None and data_buffer["depth"] is not None:
            rgb_image = data_buffer["rgb"]
            H = rgb_image.shape[0]
            W = rgb_image.shape[1]
            bilateral = cv2.bilateralFilter(rgb_image, d=9, sigmaColor=75, sigmaSpace=75)
            gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 100)
            edges = cv2.dilate(edges, None, iterations=1)
            hough_image = np.zeros_like(rgb_image)

            # Detect lines using HoughLinesP
            image_lines = cv2.HoughLinesP(edges, 1, 2*np.pi/180, threshold=50, minLineLength=10, maxLineGap=20)
            if image_lines is not None:
                image_lines = np.array(image_lines[:, 0])
            image_lines = image_lines[np.logical_and(image_lines[:, 3] > (2 * H // 3),image_lines[:, 1] > H//3)]
            image_lines = image_lines.reshape(-1, 2, 2)
                                                     
            cv2.polylines(hough_image, image_lines, isClosed=False, color=(0, 255, 0), thickness=2)

            depth_image = data_buffer["depth"]
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_canny = cv2.Canny(depth_normalized, 100, 200)
            depth_canny = cv2.dilate(depth_canny, None, iterations=1)
            hough_depth = np.zeros_like(rgb_image)
            depth_lines = cv2.HoughLinesP(depth_canny, 1, 2*np.pi/180, threshold=100, minLineLength=20, maxLineGap=10)

            # Draw lines on the image
            if depth_lines is not None:
                for line in depth_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(hough_depth, (x1, y1), (x2, y2), (0, 255, 0), 2)

            segments, scores = pyelsed.detect(gray)
            # # merged_segments = merge_lines(segments, distance_threshold=5)

            elsed_image = np.zeros_like(rgb_image)

            if segments is not None:
                for x1, y1, x2, y2 in segments:
                    slope = (y2 - y1) / (x2 - x1)
                    if np.linalg.norm([x1 - x2, y1 - y2]) > 5:
                        if np.abs(slope) > 5:
                            cv2.line(elsed_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        else:
                            cv2.line(elsed_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green lines

            # rgb_image_with_depth = cv2.addWeighted(rgb_image, 0.7, cv2.cvtColor(depth_contours, cv2.COLOR_GRAY2BGR), 0.3, 0)

            # Show the combined image
            cv2.imshow("ELSED Lines", elsed_image)
            cv2.imshow("image", rgb_image)
            cv2.imshow("Hough Lines", hough_image)
            cv2.imshow("Hough Depth", hough_depth)
            cv2.imshow("RGB Canny", edges)
            cv2.imshow("Depth Canny", depth_canny)
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