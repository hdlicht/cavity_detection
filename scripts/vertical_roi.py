#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image

from inference import get_model
import supervision as sv
import cv2
from cv_bridge import CvBridge
import threading
import numpy as np
import threading
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from ransac import ransac_plane_fitting
from geometry_msgs.msg import Point, Quaternion
from cavity_detection.msg import Roi
import tf


# Global shared buffer
data_buffer = {"rgb": None, "depth": None}
buffer_lock = threading.Lock()
tf_listener = None 
time_stamp = None
model = None

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
    [0,  0,  1,  0],  
    [-1,  0, 0,  0],  
    [0,  -1,  0,  0.6069], 
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
            results = model.infer(rgb_image)[0]
            detections = sv.Detections.from_inference(results).xyxy
            if len(detections) < 3:
                print("Not enough detections")
                return
            pixels = get_pixels_in_rectangles(detections)
            points_3d = get_3d_points(depth_image, pixels)
            
            header = Header()
            header.stamp = time_stamp
            header.frame_id = "base_footprint"
            T = T_camera_world #tf_listener.asMatrix("map", header)

            points_3d = np.column_stack((points_3d, np.ones(len(points_3d))))
            points_3d = np.dot(T, points_3d.T).T[:, :3]

            plane_model, inliers = ransac_plane_fitting(points_3d, distance_threshold=0.02, num_iterations=500)
            a, b, c, d = plane_model
            inlier_pixels = pixels[inliers]
            inlier_points = points_3d[inliers]

            min_x, min_y, min_z = np.min(inlier_points, axis=0)
            max_x, max_y, max_z = np.max(inlier_points, axis=0)

            # Publish the vertical ROI
            vert_roi = Roi()
            vert_roi.header = header
            vert_roi.roi_type = 1
            vert_roi.center = Point((min_x + max_x) / 2, (min_y + max_y) / 2, 0)
            vert_roi.width = ((max_x - min_x)**2 + (max_y - min_y)**2)**0.5
            vert_roi.depth = max_z - min_z
            x = np.array([a, b, 0])
            x = x / np.linalg.norm(x)
            z = np.array([0, 0, 1])
            y = np.cross(z, x)
            y = y / np.linalg.norm(y)
            R = np.column_stack((x, y, z))
            T = np.eye(4)
            T[:3, :3] = R
            q = tf.transformations.quaternion_from_matrix(T)
            vert_roi.orientation = Quaternion(q[0], q[1], q[2], q[3])

            vert_pub.publish(vert_roi)

            # Draw the inliers on the image

            min_x, min_y = np.min(inlier_pixels, axis=0)
            max_x, max_y = np.max(inlier_pixels, axis=0)

            for x, y in inlier_pixels:
                rgb_image[y, x] = (0, 255, 0)

            cv2.rectangle(rgb_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

            for detection in detections:
                x1, y1, x2, y2 = detection.astype(int)
                height = y2 - y1
                X, Y = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
                for x, y in zip(X.ravel(), Y.ravel()):
                    if y+height >= H:
                        continue
                    rgb_image[y, x] = rgb_image[y + height, x]

            pub.publish(bridge.cv2_to_imgmsg(rgb_image, encoding="bgr8"))

        else:
            rospy.logwarn("Waiting for both RGB and Depth images to be received.")

def rgb_callback(msg):
    """Callback function for RGB images."""
    global time_stamp
    try:
        rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with buffer_lock:
            data_buffer["rgb"] = rgb_image
            time_stamp = msg.header.stamp
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
    rospy.init_node("vertical_detector_node")
    bridge = CvBridge()
    tf_listener = tf.TransformListener()
    model = get_model("ct-logo/1")

    # Subscribe to RGB and Depth topics
    video_topic = "/camera/rgb/image_raw"
    depth_topic = "/camera/depth/image"
    rospy.Subscriber(video_topic, Image, rgb_callback, queue_size=10)
    rospy.Subscriber(depth_topic, Image, depth_callback, queue_size=10)
    pub = rospy.Publisher('/new_image', Image, queue_size=10)
    vert_pub = rospy.Publisher('/vert_roi', Roi, queue_size=10)
    marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)


    # Timer to call process_fusion() periodically (e.g., every 100ms)
    rospy.Timer(rospy.Duration(1.0 / FUSION_RATE), process_fusion)

    rospy.spin()