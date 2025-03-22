import open3d as o3d
import rospy
import numpy as np
import cv2
import threading
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Global shared buffer
data_buffer = {"depth": None, "edges": None}
buffer_lock = threading.Lock()
RUN_RATE = 5  # Reduce frequency to 5 Hz
bridge = CvBridge()

def process_depth(depth_image):
    """Process depth image to extract 3D edges."""
    depth_image = (depth_image / 1000.).astype(np.float32)
    o3d_depth = o3d.geometry.Image(depth_image)

    # Camera intrinsics (adjust based on calibration)
    fx, fy = 500, 500  
    cx, cy = depth_image.shape[1] // 2, depth_image.shape[0] // 2  
    intrinsics = o3d.camera.PinholeCameraIntrinsic(depth_image.shape[1], depth_image.shape[0], fx, fy, cx, cy)

    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, intrinsics)

    # Estimate normals efficiently
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    
    # Compute curvature (change in normal direction)
    normals = np.asarray(pcd.normals)
    print(normals.shape)
    curvature = np.linalg.norm(np.gradient(normals, axis=0), axis=1)

    # Extract edges
    edges = (curvature > np.percentile(curvature, 90)).astype(np.uint8) * 255

    # Store edges safely for rendering in the main thread
    with buffer_lock:
        data_buffer["edges"] = edges

def detect(event):
    """Periodically process depth data in a separate thread."""
    with buffer_lock:
        if data_buffer["depth"] is None:
            return  
        depth_image = data_buffer["depth"].copy()  
        data_buffer["depth"] = None  # Process only new data

    threading.Thread(target=process_depth, args=(depth_image,), daemon=True).start()

def depth_callback(msg):
    """Store depth image in a shared buffer."""
    try:
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with buffer_lock:
            data_buffer["depth"] = depth_image
    except Exception as e:
        rospy.logerr(f"Error in Depth callback: {e}")

def show_edges():
    """Display processed edges in the main thread."""
    while not rospy.is_shutdown():
        with buffer_lock:
            edges = data_buffer["edges"]
        if edges is not None:
            cv2.imshow("3D Edges", edges)
            cv2.waitKey(1)

if __name__ == "__main__":
    rospy.init_node("horizontal_detector_node")

    depth_topic = "/camera/aligned_depth_to_color/image_raw"
    rospy.Subscriber(depth_topic, Image, depth_callback, queue_size=1)

    rospy.Timer(rospy.Duration(1.0 / RUN_RATE), detect)  # Lower frequency
    rospy.spin()
    # # Run OpenCV display in the main thread
    show_edges()
