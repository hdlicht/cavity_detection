#!/usr/bin/env python

import rospy
import rosbag
from rospy.rostime import Time
from std_msgs.msg import Header

def increment_timestamps(bag_file, output_bag_file):
    # Open the input bag file
    with rosbag.Bag(bag_file, 'r') as in_bag:
        # Open the output bag file
        with rosbag.Bag(output_bag_file, 'w') as out_bag:
            # Get the start time of the bag
            start_time = in_bag.get_start_time()
            
            # Loop indefinitely
            for i in range(10):
                # Loop through each message in the bag
                for topic, msg, t in in_bag.read_messages():
                    # Increment the timestamp by the time difference since the start
                    new_timestamp = Time.from_sec(t.to_sec() + (rospy.Time.now().to_sec() - start_time))
                    
                    # Update the message header timestamp if it exists
                    if hasattr(msg, 'header') and isinstance(msg.header, Header):
                        msg.header.stamp = new_timestamp
                    
                    # Write the updated message to the output bag
                    out_bag.write(topic, msg, new_timestamp)
                
                # Reset the start time for the next loop
                start_time = rospy.Time.now().to_sec()

if __name__ == "__main__":
    # Initialize the ROS node
    rospy.init_node('increment_timestamps')
    
    # Specify the input and output bag files
    input_bag_file = 'bags/moving1.bag'
    output_bag_file = 'bags/moving4.bag'
    
    try:
        # Call the function to increment timestamps
        increment_timestamps(input_bag_file, output_bag_file)
    except rospy.ROSInterruptException:
        pass