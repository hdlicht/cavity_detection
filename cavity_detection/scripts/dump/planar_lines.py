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
import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
from visualization_msgs.msg import Marker, MarkerArray
from cavity_detection.scripts.dump.pcl_helper import ros_to_pcl, pcl_to_ros  # Helper functions for PointCloud2



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
    [0,  0,  1,  0],  
    [0, -1,  0,  0.6069], 
    [0,  0,  0,  1] ])

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

    # # Apply camera transformation
    # point = np.array([x, y, z, 1])
    # point = T_camera_world @ point
    # x, y, z = point[:3]

    return np.array([x, y, z])

def get_inliers(points, plane_model, threshold=0.03):
    """Get the inliers of a plane model."""
    a, b, c, d = plane_model
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    distances = np.abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)
    return np.where(distances < threshold)[0]

def calibrate_floor()

def process_fusion(event):
    """Periodically process the fusion of RGB and Depth images."""
    global T_depth_rgb, T_camera_world

    with buffer_lock:
        if data_buffer["rgb"] is not None and data_buffer["depth"] is not None:
            rgb_image = data_buffer["rgb"]
            depth_image = data_buffer["depth"]
            H, W = rgb_image.shape[:2]
            bilateral = cv2.bilateralFilter(rgb_image, d=9, sigmaColor=75, sigmaSpace=75)
            gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
            rgb_edges = cv2.Canny(gray, 40, 80)

            # draw a rectangle in the bottom 30 pixels of the image
            rgb_edges = cv2.cvtColor(rgb_edges, cv2.COLOR_GRAY2BGR)
            bottom = [0, H//2, W, H]
            top = [0, 0, W, H//2]
            # convert all pixels in the rectangle to 3d
            floor_points = []
            floor_points_2d = []
            for y in range(bottom[1], bottom[3]):
                for x in range(bottom[0], bottom[2]):
                    floor_points_2d.append([x, y])
                    point = get_3d_point(depth_image, x, y)
                    if point is not None:
                        floor_points.append(point)
            floor_points = np.array(floor_points)
            floor_points_2d = np.array(floor_points_2d)

            floor_pcd = o3d.geometry.PointCloud()
            floor_pcd.points = o3d.utility.Vector3dVector(floor_points)
            # Fit a plane to the points with RANSAC
            floor_model, inliers = floor_pcd.segment_plane(distance_threshold=0.03, ransac_n=10, num_iterations=1000)
            [a, b, c, d] = floor_model

            # Normalize the plane normal
            plane_normal = np.array([a, b, c])
            plane_normal /= np.linalg.norm(plane_normal)

            # Target normal vector (aligned with z-axis)
            target_normal = np.array([0, 0, -1])

            # Compute the rotation axis and angle
            rotation_axis = np.cross(plane_normal, target_normal)
            rotation_angle = np.arccos(np.dot(plane_normal, target_normal))

            # Create the rotation matrix using Rodrigues' rotation formula
            if np.linalg.norm(rotation_axis) > 1e-6:  # Avoid division by zero for parallel vectors
                rotation_axis /= np.linalg.norm(rotation_axis)
                K = np.array([
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]
                ])
                rotation_matrix = (
                    np.eye(3) +
                    np.sin(rotation_angle) * K +
                    (1 - np.cos(rotation_angle)) * (K @ K)
                )
            else:
                # No rotation needed if the plane is already aligned
                rotation_matrix = np.eye(3)

            floor_offset = d / np.linalg.norm(plane_normal)  # Distance of the plane from origin

            # Get homogeneous transformation matrix
            T = np.eye(4)
            T[:3, :3] = rotation_matrix
            T[:3, 3] = floor_offset * rotation_matrix @ plane_normal

            # print the transformation matrix rounded to 3 decimal places
            print(f'Transformation matrix:\n{T.round(3)}')

            floor_points = np.asarray(floor_pcd.points)

            # Remove inliers
            floor_points = np.delete(floor_points, inliers, axis=0)
        

            cv2.rectangle(rgb_edges, (bottom[0], bottom[1]), (bottom[2], bottom[3]), [255, 0, 0], 2)
            cv2.imshow("grayscale image", gray)
            cv2.imshow("RGB Edges", rgb_edges)
            cv2.imshow("Depth lines", depth_image)
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