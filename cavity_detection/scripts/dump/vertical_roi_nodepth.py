#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge
import threading
import numpy as np
import threading
from std_msgs.msg import Header, Int16MultiArray
from visualization_msgs.msg import Marker
from ransac import ransac_plane_fitting
from geometry_msgs.msg import Point, Quaternion
from cavity_detection_msgs.msg import Roi
import tf


# Global shared buffer
data_buffer = {"rgb": None, "depth": None}
buffer_lock = threading.Lock()
tf_listener = None 
time_stamp = None
model = None

# Desired frequency (Hz)
RUN_RATE = 10  # 10 times per second (i.e., 100 ms period)

K_rgb = np.array([[570.342, 0.0,     314.5], 
                [0.0,     570.342, 235.5],
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

def normalize_lines(lines):
    # Ensure lines have consistent ordering: (x1, y1, x2, y2) where x1 < x2 or (x1 == x2 and y1 < y2)
    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
    swap = (x1 > x2) | ((x1 == x2) & (y1 > y2))
    normalized = lines.copy()
    normalized[swap] = lines[swap][:, [2, 3, 0, 1]]
    return normalized

def process_fusion(event):
    """Periodically process."""
    with buffer_lock:
        if data_buffer["rgb"] is not None:
            rgb_image = data_buffer["rgb"]
            H, W = rgb_image.shape[:2]
            detections = ## DETECTION MODEL HERE
            if len(detections) < 3:
                return

            # Merge detections            
            reduced_detections = []
            #sort detections by x1
            detections = np.array(detections)
            detections = np.argsort(detections, axis=0)
            min_x, min_y = detections[0][0], detections[0][1]
            max_x, max_y = detections[0][2], detections[0][3]
            for i in range(1, len(detections)):
                x1, y1 = detections[i][0], detections[i][1]
                x2, y2 = detections[i][2], detections[i][3]
                if x2 < max_x:
                    max_x = x
                    if y1 < min_y:
                        min_y = y1
                    if y2 > max_y:
                        max_y = y2
                else:
                    reduced_detections.append([min_x, min_y, max_x, max_y])
                    min_x, min_y = x1, y1
                    max_x, max_y = x2, y2
            reduced_detections.append([min_x, min_y, max_x, max_y])
            reduced_detections = np.array(reduced_detections).dtype(np.int32)

            # Publish the vertical ROI

            # vert_roi = Roi()
            # header = Header()
            # header.stamp = time_stamp
            # header.frame_id = "base_footprint"
            # vert_roi.header = header
            # vert_roi.roi_type = 1
            # vert_roi.center = Point((min_x + max_x) / 2, (min_y + max_y) / 2, 0)
            # vert_roi.width = ((max_x - min_x)**2 + (max_y - min_y)**2)**0.5
            # vert_roi.depth = max_z
            # x = np.array([a, b, 0])
            # x = x / np.linalg.norm(x)
            # z = np.array([0, 0, 1])
            # y = np.cross(z, x)
            # y = y / np.linalg.norm(y)
            # R = np.column_stack((x, y, z))
            # T = np.eye(4)
            # T[:3, :3] = R
            # q = tf.transformations.quaternion_from_matrix(T)
            # vert_roi.orientation = Quaternion(q[0], q[1], q[2], q[3])
            # vert_pub.publish(vert_roi)


            msg = Int16MultiArray()
            for detection in reduced_detections:
                min_x, min_y, max_x, max_y = detection
                msg.data.extend(detection)
            pub2.publish(msg)

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
    model = ## LOAD YOUR MODEL HERE

    # Subscribe to RGB and Depth topics
    video_topic = "/camera/rgb/image_raw"
    rospy.Subscriber(video_topic, Image, rgb_callback, queue_size=10)
    pub2 = rospy.Publisher("/int_pair", Int16MultiArray, queue_size=10)
    vert_pub = rospy.Publisher('/vert_roi', Roi, queue_size=10)



    # Timer to call process_fusion() periodically (e.g., every 100ms)
    rospy.Timer(rospy.Duration(1.0 / RUN_RATE), process_fusion)

    rospy.spin()