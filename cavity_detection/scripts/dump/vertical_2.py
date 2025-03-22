#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image

from inference import get_model
import supervision as sv
import cv2
from cv_bridge import CvBridge
import threading
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import threading
from std_msgs.msg import Float32MultiArray
import open3d as o3d

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

def get_3d_points(depth_image, points_2d):
    """Get the 3D points for an array of 2D pixels using the depth image."""
    zs = depth_image[points_2d[:, 1], points_2d[:, 0]]
    valid = zs > 0  # Filter out invalid depth values
    xs = (points_2d[:, 0] - K_rgb[0, 2]) / K_rgb[0, 0] * zs
    ys = (points_2d[:, 1] - K_rgb[1, 2]) / K_rgb[1, 1] * zs
    points_3d = np.vstack((xs, ys, zs)).T
    return points_3d[valid]

def normalize_lines(lines):
    # Ensure lines have consistent ordering: (x1, y1, x2, y2) where x1 < x2 or (x1 == x2 and y1 < y2)
    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
    swap = (x1 > x2) | ((x1 == x2) & (y1 > y2))
    normalized = lines.copy()
    normalized[swap] = lines[swap][:, [2, 3, 0, 1]]
    return normalized

import numpy as np

def get_pixels_in_rectangles(rectangles):
    """
    Generate a list of pixel locations for all given rectangles.
    
    Args:
        rectangles (list of tuples): Each rectangle is defined as (x1, y1, x2, y2).
    
    Returns:
        List of tuples: Each tuple is a pixel location (x, y) that lies within the rectangles.
    """
    pixels = []
    for x1, y1, x2, y2 in rectangles:
        # Create a grid of pixel locations for the current rectangle
        ys, xs = np.meshgrid(np.arange(y1, y2), np.arange(x1, x2), indexing="ij")
        # Flatten and combine the grid into pixel locations
        rect_pixels = np.column_stack((xs.ravel(), ys.ravel()))
        pixels.append(rect_pixels)

    pixels = np.vstack(pixels).astype(int)
    # remove duplicates
    pixels = np.unique(pixels, axis=0)
    
    return pixels

def process_fusion(event):
    """Periodically process the fusion of RGB and Depth images."""
    with buffer_lock:
        if data_buffer["rgb"] is not None and data_buffer["depth"] is not None:
            rgb_image = data_buffer["rgb"]
            inlier_image = rgb_image.copy()
            depth_image = data_buffer["depth"]
            H, W = rgb_image.shape[:2]
            model = get_model("ct-logo/1")
            results = model.infer(rgb_image)[0]
            detections = sv.Detections.from_inference(results).xyxy
            print(detections.shape)
            if len(detections) == 0:
                rospy.logwarn("No detections found.")
                return
            # height = detections[:, 3] - detections[:, 1]
            # width = detections[:, 2] - detections[:, 0]
            # expander = np.vstack((-width, -height, width, height)).T
            # big_detections = detections + expander
            # big_detections = np.clip(big_detections, 0, [W, H, W, H])
            pixels = get_pixels_in_rectangles(detections)
            points_3d = get_3d_points(depth_image, pixels)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3d)
            # Fit a plane to the points with RANSAC
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=500)
            # draw the inlier points on the rgb image
            inlier_pixels = pixels[inliers]
            rect = cv2.minAreaRect(inlier_pixels)
            # draw the rectangle on the rgb image
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(inlier_image, [box], 0, (0, 255, 0), 2)

                

            for detection in detections:
                x1, y1, x2, y2 = detection.astype(int)
                height = y2 - y1
                X, Y = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
                for x, y in zip(X.ravel(), Y.ravel()):
                    if y+height >= H:
                        continue
                    rgb_image[y, x] = rgb_image[y + height, x]
                # cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            pub.publish(bridge.cv2_to_imgmsg(rgb_image, encoding="bgr8"))
            pub2.publish(bridge.cv2_to_imgmsg(inlier_image, encoding="bgr8"))
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
    pub = rospy.Publisher('/new_image', Image, queue_size=10)
    pub2 = rospy.Publisher('/inlier_image', Image, queue_size=10)


    # Timer to call process_fusion() periodically (e.g., every 100ms)
    rospy.Timer(rospy.Duration(1.0 / FUSION_RATE), process_fusion)

    rospy.spin()