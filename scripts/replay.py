#!/usr/bin/env python3

import rospy
import rosbag
from std_msgs.msg import Header
from rosgraph_msgs.msg import Clock
from rospy import Time
import sys

bag_file = ""

def republish_with_continuous_time():
    rospy.init_node("bag_timestamp_modifier", anonymous=True)

    publishers = {}

    # Open the bag for reading
    with rosbag.Bag(bag_file, "r") as bag:
        start_time = bag.get_start_time()
        last_t = rospy.Time.from_sec(start_time)
        while not rospy.is_shutdown():
            for topic, msg, t in bag.read_messages():
                # sleep to maintain the loop rate
                rospy.sleep(t - last_t)

                # Create publishers for each topic if not already created
                if topic not in publishers:
                    publishers[topic] = rospy.Publisher(topic, type(msg), queue_size=10)

                if rospy.is_shutdown():
                    break

                # Adjust the message's header timestamp, if present
                if hasattr(msg, "header"):
                    msg.header.stamp = rospy.Time.now()

                
                # If message is TF, adjust the timestamp of the transform
                if topic == "/tf" or topic == "/tf_static":
                    for transform in msg.transforms:
                        transform.header.stamp = rospy.Time.now()
                # Publish the message
                publishers[topic].publish(msg)

                # Set the last timestamp to the current timestamp
                last_t = t

            # Update time offset for next loop
            last_t = rospy.Time.from_sec(start_time)
            

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            bag_file = sys.argv[1]
        else:
            rospy.logerr("Please provide the bag file path as a command line argument.")
            sys.exit(1)
        republish_with_continuous_time()
    except rospy.ROSInterruptException:
        pass