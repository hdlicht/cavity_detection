import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber

def callback(rgb_msg, depth_msg):
    try:
        # Convert ROS messages to OpenCV images
        rgb_image = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        # Normalize depth for visualization
        if depth_image.dtype == np.uint16:
            depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif depth_image.dtype == np.float32:
            depth_normalized = cv2.normalize(depth_image, None, 0, 1.0, cv2.NORM_MINMAX) * 255
            depth_normalized = depth_normalized.astype(np.uint8)
        else:
            rospy.logwarn("Unexpected depth image format")
            return

        # Apply a colormap for depth visualization
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # Display the RGB and depth images
        cv2.imshow("RGB Feed", rgb_image)
        cv2.imshow("Depth Feed", depth_colormap)
        cv2.waitKey(1)
    except Exception as e:
        rospy.logerr(f"Error processing images: {e}")

if __name__ == "__main__":
    rospy.init_node("image_depth_viewer")
    bridge = CvBridge()

    # Subscribers for the two topics
    rgb_sub = Subscriber("/camera/rgb/image_raw", Image)
    depth_sub = Subscriber("/camera/depth/image", Image)

    # Synchronize the two topics
    ats = ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.1)
    ats.registerCallback(callback)

    rospy.loginfo("Synchronized subscribers are ready")
    rospy.spin()
    cv2.destroyAllWindows()
