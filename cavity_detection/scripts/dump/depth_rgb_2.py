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



# Global shared buffer
data_buffer = {"rgb": None, "depth": None}
buffer_lock = threading.Lock()
pub = None

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

def process_fusion(event):
    """Periodically process the fusion of RGB and Depth images."""
    global pub
    with buffer_lock:
        if data_buffer["rgb"] is not None and data_buffer["depth"] is not None:
            rgb_image = data_buffer["rgb"]
            H, W = rgb_image.shape[:2]
            bilateral = cv2.bilateralFilter(rgb_image, d=9, sigmaColor=75, sigmaSpace=75)
            gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
            rgb_edges = cv2.Canny(gray, 40, 80)
            #for i in range(1):
                #rgb_edges = cv2.dilate(rgb_edges, None, iterations=1)
                #rgb_edges = cv2.erode(rgb_edges, None, iterations=1)

            depth_image = data_buffer["depth"]
            too_far = np.isnan(depth_image) # Filter out values too far or NaN
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_edges = cv2.Canny(depth_normalized, 30, 50)
            # remove edges in top half of the image
            depth_edges[:H//2, :] = 0
            # depth_edges[too_far] = 0
            # rgb_edges[too_far] = 0
            # depth_edges = cv2.dilate(depth_edges, None, iterations=1)
            # depth_edges = cv2.erode(depth_edges, None, iterations=1)

            # Find contours
            closed_rgb = cv2.morphologyEx(rgb_edges, cv2.MORPH_CLOSE, kernel=np.ones((1, 1), np.uint8))
            closed_depth = cv2.morphologyEx(depth_edges, cv2.MORPH_CLOSE, kernel=np.ones((7, 7), np.uint8))
            rgb_contours, _ = cv2.findContours(closed_rgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            depth_contours, _ = cv2.findContours(closed_depth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Generate random colors for each contour
            colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(depth_contours))]
            depth_normalized = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
            triangle_image = depth_normalized.copy()
            all_planes = []

            # Draw each contour with a unique color
            for i, contour in enumerate(depth_contours):
                if len(contour) < 300 or len(contour) > 1000:
                    continue
                length = cv2.arcLength(contour, closed=True)

                cv2.drawContours(depth_normalized, [contour], -1, colors[i], thickness=2)
                    # Approximate shape (simplify noisy contours)

                epsilon = 0.01 * length  # Tolerance for approximation
                approx = cv2.approxPolyDP(contour, epsilon, closed=True)

                # Find minimal enclosing triangle
                _, triangle = cv2.minEnclosingTriangle(approx)
                triangle = np.int32(triangle)

                # Draw the triangle
                cv2.polylines(triangle_image, [triangle], isClosed=True, color=(0, 255, 0), thickness=2)

                # Get a list of points that are inside the triangle
                points = []

                for y in range(H):
                    for x in range(W):
                        if cv2.pointPolygonTest(triangle, (x, y), measureDist=False) >= 0:
                            point = get_3d_point(depth_image, x, y)
                            if point is not None:
                                points.append(point)
                points = np.array(points)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                # Fit a plane to the points with RANSAC
                if len(points) > 10:
                    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=500)
                    [a, b, c, d] = plane_model
                    plane_data = np.array([a, b, c, d, 3, .3])
                    print(f"Plane model: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
                    # if plane is vertical, publish the plane model coefficients
                    if abs(c) < 0.2:
                        all_planes.append(plane_data)
            if len(all_planes) > 0:
                all_planes = np.array(all_planes)
                pub.publish(Float32MultiArray(data=all_planes.flatten()))

            # Draw each contour with a unique color
            colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(rgb_contours))]
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            for i, contour in enumerate(rgb_contours):
                if len(contour) < 10:
                    continue
                cv2.drawContours(gray, [contour], -1, colors[i], thickness=2)                


            # display the contours
            # cv2.drawContours(rgb_image, rgb_contours, -1, (0, 255, 0), 2)
            # cv2.drawContours(depth_normalized, depth_contours, -1, (0, 255, 0), 2)

            # Visualize corners
            # Harris corner detection
            harris_corners = cv2.cornerHarris(rgb_edges, blockSize=3, ksize=21, k=0.1)

            # Threshold to identify strong corners
            corners = np.zeros_like(rgb_edges)
            corners[harris_corners > 0.01 * harris_corners.max()] = 255
            output = cv2.cvtColor(rgb_edges, cv2.COLOR_GRAY2BGR)
            output[corners > 0] = [0, 0, 255]  # Mark corners in red

            # cv2.imshow("Harris Corners", output)
            cv2.imshow("RGB Contours", gray)
            cv2.imshow("Depth Contours", depth_normalized)
            cv2.imshow("Output", triangle_image)
            cv2.imshow("RGB Edges", rgb_edges)
            cv2.imshow("Depth lines", depth_edges)
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