import rospy
import tf
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

# Initialize ROS node
rospy.init_node('align_depth_rgb')

# Global variables
bridge = CvBridge()
rgb_image = None
depth_image = None
rgb_info = None
depth_info = None
tf_listener = tf.TransformListener()

def camera_info_callback(data, camera_type):
    global rgb_info, depth_info
    if camera_type == "rgb":
        rgb_info = data
    elif camera_type == "depth":
        depth_info = data

def image_callback(data, camera_type):
    global rgb_image, depth_image
    if camera_type == "rgb":
        rgb_image = bridge.imgmsg_to_cv2(data, "bgr8")
    elif camera_type == "depth":
        depth_image = bridge.imgmsg_to_cv2(data, "passthrough")

# Subscribers
rospy.Subscriber('/camera/rgb/image_raw', Image, image_callback, "rgb")
rospy.Subscriber('/camera/depth/image', Image, image_callback, "depth")
rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, camera_info_callback, "rgb")
rospy.Subscriber('/camera/depth/camera_info', CameraInfo, camera_info_callback, "depth")

rospy.sleep(1)  # Wait for messages

# Ensure data is received
if rgb_image is None or depth_image is None or rgb_info is None or depth_info is None:
    rospy.logerr("Missing data!")
    exit(1)

# Intrinsic matrices
K_rgb = np.array(rgb_info.K).reshape(3, 3)
K_depth = np.array(depth_info.K).reshape(3, 3)

# Transform between depth and RGB
try:
    trans, rot = tf_listener.lookupTransform(rgb_info.header.frame_id, depth_info.header.frame_id, rospy.Time(0))
    T = tf.transformations.compose_matrix(translate=trans, angles=tf.transformations.euler_from_quaternion(rot))
except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    rospy.logerr("Transform not available!")
    exit(1)

# depth intrinsics
h_depth, w_depth = depth_image.shape
fx_depth, fy_depth = K_depth[0, 0], K_depth[1, 1]
cx_depth, cy_depth = K_depth[0, 2], K_depth[1, 2]

# rgb intrinsics
h_rgb, w_rgb = rgb_image.shape[:2]
fx_rgb, fy_rgb = K_rgb[0, 0], K_rgb[1, 1]
cx_rgb, cy_rgb = K_rgb[0, 2], K_rgb[1, 2]

while not rospy.is_shutdown():
# Find 3D points from depth image
    points_3d = []
    for v in range(h_depth):
        for u in range(w_depth):
            z = depth_image[v, u]  # Depth value
            if z == 0:  # Skip invalid depth
                continue
            x = (u - cx_depth) * z / fx_depth
            y = (v - cy_depth) * z / fy_depth
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                continue
            points_3d.append([x, y, z])


    # transform points to RGB frame
    points_3d = np.array(points_3d)
    points_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    points_3d = np.dot(points_3d, T.T)
    points_3d = points_3d[:, :3]    
    print(points_3d.shape)

    # find depth of each point in RGB frame
    points_2d = []
    for i in range(points_3d.shape[0]):
        x, y, z = points_3d[i]
        u = int(fx_rgb * x / z + cx_rgb)
        v = int(fy_rgb * y / z + cy_rgb)
        points_2d.append([u, v])
    aligned_depth = np.zeros((h_rgb, w_rgb), dtype=np.float32)
    for i in range(len(points_2d)):
        u, v = points_2d[i]
        if 0 <= u < w_rgb and 0 <= v < h_rgb:
            aligned_depth[v, u] = points_3d[i, 2]

    nans = np.isnan(depth_image)
    cv2.imshow("Nans in Depth", nans.astype(np.uint8) * 255)
    cv2.imshow("Aligned Depth", aligned_depth)
    cv2.imshow("RGB Image", rgb_image)
    cv2.waitKey(1)