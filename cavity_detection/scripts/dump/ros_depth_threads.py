import queue
import threading
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

frame_queue = queue.Queue(maxsize=10)  # Shared queue for frames

def display_thread():
    while not rospy.is_shutdown():
        try:
            window_name, frame = frame_queue.get(timeout=1)  # Wait for frames
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)
        except queue.Empty:
            continue
    cv2.destroyAllWindows()

def image_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if frame_queue.full():
            frame_queue.get()  # Drop the oldest frame
        frame_queue.put(("Video Feed", cv_image))
    except Exception as e:
        rospy.logerr(f"Error in image callback: {e}")

def depth_callback(msg):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put(("Depth Feed", cv_image))
    except Exception as e:
        rospy.logerr(f"Error in depth callback: {e}")

if __name__ == "__main__":
    rospy.init_node("video_feed_viewer")
    bridge = CvBridge()

    # Start display thread
    threading.Thread(target=display_thread, daemon=True).start()

    # Subscribe to topics
    video_topic = "/camera/rgb/image_raw"
    depth_topic = "/camera/depth/image"
    rospy.Subscriber(video_topic, Image, image_callback, queue_size=10)
    rospy.Subscriber(depth_topic, Image, depth_callback, queue_size=10)

    rospy.spin()
