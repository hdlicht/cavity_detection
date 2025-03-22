import open3d as o3d
import rospy
import numpy as np
import cv2
import threading
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Global shared buffer
data_buffer = {"rgb": None, "depth": None}
buffer_lock = threading.Lock()
RUN_RATE = 5 

def detect(event):
    """Periodically process the fusion of RGB and Depth images."""
    global data_buffer, buffer_lock, pub, models, calibration_count, average_model
    with buffer_lock:
        if data_buffer["depth"] is not None:
            depth_image = data_buffer["depth"]
            depth_image = (depth_image/1000.).astype(np.float32)
            o3d_depth = o3d.geometry.Image(depth_image)


            # Create a point cloud from depth image
            fx, fy = 500, 500  # Focal length (adjust as needed)
            cx, cy = depth_image.shape[1] // 2, depth_image.shape[0] // 2  # Principal point

            intrinsics = o3d.camera.PinholeCameraIntrinsic(depth_image.shape[1], depth_image.shape[0], fx, fy, cx, cy)
            pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, intrinsics)

            # Estimate normals
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

            # Compute curvature (change in normal direction)
            normals = np.asarray(pcd.normals)
            curvature = np.linalg.norm(np.gradient(normals, axis=0), axis=1)

            # Threshold to get edges
            edges = (curvature > np.percentile(curvature, 90)).astype(np.uint8) * 255

            # Show edges
            cv2.imshow("3D Edges", edges)
            cv2.waitKey(1)


def depth_callback(msg):
    """Callback function for Depth images."""
    global time_stamp
    try:
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with buffer_lock:
            data_buffer["depth"] = depth_image
            time_stamp = msg.header.stamp
    except Exception as e:
        rospy.logerr(f"Error in Depth callback: {e}")

if __name__ == "__main__":
    rospy.init_node("horizontal_detector_node")
    bridge = CvBridge()
    depth_topic = "/camera/aligned_depth_to_color/image_raw"
    rospy.Subscriber(depth_topic, Image, depth_callback, queue_size=2)

    # Timer to call process_fusion() periodically (e.g., every 100ms)
    rospy.Timer(rospy.Duration(1.0 / RUN_RATE), detect)
    rospy.spin()