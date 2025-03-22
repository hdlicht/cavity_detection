import open3d as o3d
import numpy as np
import rospy
import threading
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# Global shared buffer
data_buffer = {"depth": None}
buffer_lock = threading.Lock()
RUN_RATE = 50  

# Known ground plane normal
GROUND_NORMAL = np.array([0, 0, 1])  
NORMAL_THRESHOLD = 0.1  # Adjust as needed

# ROS Publisher
pub = None

T_camera_world = np.array([
    [0,  0,  1,  0],  
    [-1,  0, 0,  0],  
    [0,  -1,  0,  0.7], 
    [0,  0,  0,  1] ])


def detect(event):
    """Process and filter point cloud from depth image."""
    global data_buffer, buffer_lock, pub

    with buffer_lock:
        if data_buffer["depth"] is None:
            return
        

        depth_image = (data_buffer["depth"] / 1000.0).astype(np.float32)  # Convert mm to meters
        W, H = depth_image.shape[1], depth_image.shape[0]
        depth_image = depth_image[W//2:, :]
        o3d_depth = o3d.geometry.Image(depth_image)

        # Camera Intrinsics (Adjust as needed)
        fx, fy = 910, 910  
        cx, cy = W // 2, 0 #depth_image.shape[0] // 2  
        intrinsics = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

        # Generate Point Cloud
        pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, intrinsics)
        pcd.transform(T_camera_world)

        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

        # Convert to numpy arrays
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        # Filter points that are parallel to the ground
        dot_products = np.abs(normals @ GROUND_NORMAL)  # Compute |dot(normal, ground_normal)|
        filtered_indices = np.where((dot_products < (1 - NORMAL_THRESHOLD)))[0]
        filtered_pcd = pcd.select_by_index(filtered_indices)

        # Publish to ROS
        publish_pointcloud(filtered_pcd)


def publish_pointcloud(pcd):
    """Convert Open3D point cloud to ROS PointCloud2 and publish."""
    global pub

    if pub is None:
        return

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "base_footprint"  # Adjust frame as needed

    points = np.asarray(pcd.points)
    cloud_msg = pc2.create_cloud_xyz32(header, points)

    pub.publish(cloud_msg)


def depth_callback(msg):
    """Callback function for depth images."""
    try:
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with buffer_lock:
            data_buffer["depth"] = depth_image
    except Exception as e:
        rospy.logerr(f"Depth callback error: {e}")


if __name__ == "__main__":
    rospy.init_node("plane_filter_node")
    bridge = CvBridge()

    depth_topic = "/camera/aligned_depth_to_color/image_raw"
    rospy.Subscriber(depth_topic, Image, depth_callback, queue_size=1)

    pub = rospy.Publisher("/filtered_pointcloud", PointCloud2, queue_size=1)

    rospy.Timer(rospy.Duration(1.0 / RUN_RATE), detect)
    rospy.spin()
