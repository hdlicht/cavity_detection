#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

ax = None

def line_callback(msg):
    global ax
    data = np.array(msg.data).reshape(-1, 6)  # Reshape to (N, 6): x1, y1, z1, x2, y2, z2
    ax.cla()  # Clear the previous plot

    # Plot each line segment
    for line in data:
        x1, y1, z1, x2, y2, z2 = line
        ax.plot([x1, x2], [y1, y2], [z1, z2], color='b')
    plt.draw()

if __name__ == '__main__':
    # ROS Node initialization
    rospy.init_node('line_plotting_node')

    # Subscriber to the detected lines topic
    rospy.Subscriber('/detected_3d_lines', Float32MultiArray, line_callback)

    # Set up Matplotlib 3D figure
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Continuously update the plot
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        plt.pause(0.01)  # Allow the plot to refresh
        rate.sleep()

