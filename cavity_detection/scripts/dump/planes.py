#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import threading
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import threading
from std_msgs.msg import Float32MultiArray
import open3d as o3d
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
import sys



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


models = []
average_model = None
calibration_count = 0

def normalize_lines(lines):
    # Ensure lines have consistent ordering: (x1, y1, x2, y2) where x1 < x2 or (x1 == x2 and y1 < y2)
    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
    swap = (x1 > x2) | ((x1 == x2) & (y1 > y2))
    normalized = lines.copy()
    normalized[swap] = lines[swap][:, [2, 3, 0, 1]]
    return normalized

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

def process_fusion(event):
    """Periodically process the fusion of RGB and Depth images."""
    global data_buffer, buffer_lock, pub, models, calibration_count, average_model
    with buffer_lock:
        if data_buffer["rgb"] is not None and data_buffer["depth"] is not None:
            rgb_image = data_buffer["rgb"]
            depth_image = data_buffer["depth"]
            H, W = rgb_image.shape[:2]
            bottom = [0, H//2, W, H]
            # convert all pixels in the rectangle to 3d
            x, y = np.meshgrid(range(bottom[0], bottom[2]), range(bottom[1], bottom[3]))
            points_2d = np.vstack([x.ravel(), y.ravel()]).T
            points = get_3d_points(depth_image, points_2d)
            min_xyz = np.nanmin(points, axis=0)

            if calibration_count < 50:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                # Fit a plane to the points with RANSAC
                plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=500)
                models.append(plane_model)
                calibration_count += 1
                return
            
            if calibration_count == 50:
                average_model = np.array(models).mean(axis=0)
                a, b, c, d = average_model
                print(f"Calibration done. Plane model: {a}x + {b}y + {c}z + {d} = 0")
                calibration_count += 1

            a, b, c, d = average_model
            inliers = np.where(np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) < 0.05)[0]

            # canny edge detection on the RGB image
            bilateral = cv2.bilateralFilter(rgb_image, d=9, sigmaColor=75, sigmaSpace=75)
            gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
            rgb_edges = cv2.Canny(gray, 40, 80)
            rgb_edges = cv2.cvtColor(rgb_edges, cv2.COLOR_GRAY2BGR)

            # color the inliers on the grayscale image in green
            for i in inliers:
                x, y = points_2d[i]
                rgb_edges[y, x] = [0, 255, 0]

            # get the inliers for a plane offset from the above in the z direction

            # Calculate distances of all points to the new plane
            best_inliers = []
            best_count = 0
            best_d = 0

            # find the plane parralel to the ground with the most inliers
            for delta_d in np.linspace(0.1, 0.5, 50):

                new_d = d + delta_d
                distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + new_d)

                # Identify inliers based on the distance threshold
                inlier_indices = np.where(distances < 0.01)[0]
                if len(inlier_indices) > best_count:
                    best_count = len(inlier_indices)
                    best_inliers = inlier_indices
                    best_d = delta_d

            for i in best_inliers:
                x, y = points_2d[i]
                rgb_edges[y, x] = [0, 0, 255]
            

            xy = points[best_inliers]
            xy = xy - min_xyz
            H = int(max(xy[:, 2])*100)
            W = int(max(xy[:, 0])*100)

            # Show the inliers on a blank image
            blank = np.zeros((H, W), dtype=np.uint8)
            for x, _, y in xy:
                blank[int(y*100)-1, int(x*100)-1] = 255
            blank = cv2.dilate(blank, None, iterations=2)
            blank = cv2.erode(blank, None, iterations=1)
            lines = cv2.HoughLinesP(blank, 1, np.pi/180, 100, minLineLength=70, maxLineGap=20)
            if lines is not None:
                lines = normalize_lines(lines.flatten().reshape(-1, 4))
                
            # calculate rho and theta for the lines
            # rho = np.sqrt((lines[:, 2] - lines[:, 0])**2 + (lines[:, 3] - lines[:, 1])**2)
            # theta = np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])

            # cluster the lines based on rho and theta
            # X = np.array([rho, theta]).T
            # use k means with k = 7
            #clustering = KMeans(n_clusters=7).fit(lines)
            clustering = DBSCAN(eps=10, min_samples=2).fit(lines)

            labels = clustering.labels_
            unique_labels = np.unique(labels)
            blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)
            # draw the lines in different colors based on the cluster
            # create random colors for each cluster
            colors = np.random.randint(0, 255, (len(unique_labels), 3))
            for i, label in enumerate(unique_labels):
                if label == -1:
                    continue
                indices = np.where(labels == label)[0]
                cluster_lines = lines[indices]
                min_x = np.min(cluster_lines[:, 0])
                max_x = np.max(cluster_lines[:, 2])
                min_y = np.min(cluster_lines[:, 1])
                max_y = np.max(cluster_lines[:, 3])
                color = tuple(colors[i].tolist())  # Convert color to tuple
                #cv2.line(blank, (x1, y1), (x2, y2), color, 2)
                cv2.line(blank, (min_x, min_y), (max_x, max_y), color, 2)

            # Draw the rectangle on the RGB edges image
            cv2.rectangle(rgb_edges, (bottom[0], bottom[1]), (bottom[2], bottom[3]), [255, 0, 0], 2)
            cv2.imshow("grayscale image", gray)
            cv2.imshow("RGB Edges", rgb_edges)
            cv2.imshow("Inliers", blank)
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