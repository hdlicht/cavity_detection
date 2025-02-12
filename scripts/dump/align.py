#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import threading
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from std_msgs.msg import Float32MultiArray
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
from visualization_msgs.msg import Marker, MarkerArray
from pcl_helper import ros_to_pcl, pcl_to_ros  # Helper functions for PointCloud2



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
    [1, 0, 0, 0], 
    [0, 1, 0, -0.025],
    [0, 0, 1, 0],
    [0, 0, 0, 1] ])

T_camera_world = np.array([
    [1,  0,  0,  0],  
    [0,  0,  1,  0],  
    [0, -1,  0,  0.6069], 
    [0,  0,  0,  1] ])

import numpy as np
import cv2

def map_rgb_to_depth(rgb_image, depth_image, K_rgb, K_depth, T, depth_scale=1.0):
    """
    Maps each RGB pixel to its corresponding depth pixel.

    Args:
        rgb_image (ndarray): The RGB image (HxWx3).
        depth_image (ndarray): The depth image (HxW).
        K_rgb (ndarray): Intrinsics matrix of the RGB camera (3x3).
        K_depth (ndarray): Intrinsics matrix of the depth camera (3x3).
        T (ndarray): Extrinsics matrix from RGB to depth camera (4x4).
        depth_scale (float): Scaling factor for depth values (e.g., 0.001 for millimeters to meters).

    Returns:
        ndarray: Aligned depth values for the RGB image.
    """
    # Intrinsics
    fx_rgb, fy_rgb, cx_rgb, cy_rgb = K_rgb[0, 0], K_rgb[1, 1], K_rgb[0, 2], K_rgb[1, 2]
    fx_depth, fy_depth, cx_depth, cy_depth = K_depth[0, 0], K_depth[1, 1], K_depth[0, 2], K_depth[1, 2]

    # Inverse of rotation and translation
    R = T[:3, :3]
    t = T[:3, 3]

    # Output depth map for RGB resolution
    rgb_h, rgb_w, _ = rgb_image.shape
    depth_h, depth_w = depth_image.shape
    aligned_depth = np.zeros((rgb_h, rgb_w), dtype=np.float32)

    # Create meshgrid for RGB image
    rgb_u, rgb_v = np.meshgrid(np.arange(rgb_w), np.arange(rgb_h))

    # Convert RGB pixels to 3D points in RGB camera frame
    z_rgb = np.ones_like(rgb_u).astype(np.float32)  # Set arbitrary Z=1
    x_rgb = (rgb_u - cx_rgb) / fx_rgb
    y_rgb = (rgb_v - cy_rgb) / fy_rgb

    points_rgb = np.stack([x_rgb, y_rgb, z_rgb], axis=-1).reshape(-1, 3).T  # Shape (3, N)

    # Transform points to depth camera frame
    points_depth = R @ points_rgb + t.reshape(3, 1)

    # Project points into the depth image
    z_depth = points_depth[2, :]  # Depth values in depth camera frame
    u_depth = (fx_depth * points_depth[0, :] / z_depth + cx_depth).reshape(rgb_h, rgb_w)
    v_depth = (fy_depth * points_depth[1, :] / z_depth + cy_depth).reshape(rgb_h, rgb_w)

    # Ensure coordinates are valid
    valid_mask = (u_depth >= 0) & (u_depth < depth_w) & (v_depth >= 0) & (v_depth < depth_h)
    valid_u = np.clip(u_depth, 0, depth_w - 1).astype(np.float32)
    valid_v = np.clip(v_depth, 0, depth_h - 1).astype(np.float32)
    sampled_depth = cv2.remap(depth_image, valid_u.astype(np.float32), valid_v.astype(np.float32), 
                              interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    #aligned_depth[valid_mask] = sampled_depth * depth_scale

    return sampled_depth


def get_depth(depth_image, x, y):
    """Get the depth value at pixel (x, y) from the depth image."""
    return depth_image[y, x]

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

def get_inliers(points, plane_model, threshold=0.03):
    """Get the inliers of a plane model."""
    a, b, c, d = plane_model
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    distances = np.abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)
    return np.where(distances < threshold)[0]

def process_fusion(event):
    """Periodically process the fusion of RGB and Depth images."""
    global T_depth_rgb, T_camera_world

    with buffer_lock:
        if data_buffer["rgb"] is not None and data_buffer["depth"] is not None:
            rgb_image = data_buffer["rgb"]
            depth_image = data_buffer["depth"]

            # Align depth to RGB resolution
            aligned_depth = map_rgb_to_depth(rgb_image, depth_image, K_rgb, K_depth, T_depth_rgb)
            aligned_depth = cv2.normalize(aligned_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            ad_edges = cv2.Canny(aligned_depth, 20, 80)
            ad_edges = cv2.dilate(ad_edges, None, iterations=3)
            depth_scaled = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            d_edges = cv2.Canny(depth_scaled, 20, 80)

            # convert rgb to gray
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            gauss = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(gauss, 20, 80)
            edges = cv2.dilate(edges, None, iterations=1)

            edges[ad_edges > 0] = 0
            
            # put the depth image in the blue channel
            depth_overlay = cv2.merge([edges, edges, d_edges])
            aligned_depth_overlay = cv2.merge([edges, edges, ad_edges])

            cv2.imshow("Depth Overlay", depth_overlay)
            cv2.imshow("Aligned Depth Overlay", aligned_depth_overlay)
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
    rospy.init_node("detect_planes_node")
    bridge = CvBridge()
    
    # Subscribe to RGB and Depth topics
    video_topic = "/camera/rgb/image_raw"
    depth_topic = "/camera/depth/image"
    rospy.Subscriber(video_topic, Image, rgb_callback, queue_size=10)
    rospy.Subscriber(depth_topic, Image, depth_callback, queue_size=10)
    pub = rospy.Publisher('/detected_shapes', Float32MultiArray, queue_size=10)
    
    # Timer to call process_fusion() periodically (e.g., every 100ms)
    rospy.Timer(rospy.Duration(1.0 / FUSION_RATE), process_fusion)

    rospy.spin()