import os
import cv2
import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

# Configuration
bag_file = "2025-01-17-16-04-45.bag"  # Path to your ROS bag file
image_topic = "/camera/rgb/image_raw/compressed"       # Topic containing images
output_dir = "output_images_3"            # Directory to save images

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Initialize CVBridge
bridge = CvBridge()
count = 0
# Read the ROS bag
with rosbag.Bag(bag_file, "r") as bag:
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        if topic == image_topic:
            # Convert ROS Image message to OpenCV image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # Generate filename with timestamp
            timestamp = t.to_sec()
            filename = os.path.join(output_dir, f"{timestamp:.6f}.jpg")
            # Save the image using OpenCV
            if count % 20 == 0:
                cv2.imwrite(filename, cv_image)
                print(f"Saved image to {filename}")
            count += 1