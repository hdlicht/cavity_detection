#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def line_callback(msg):
    marker = Marker()
    marker.header.frame_id = "map"  # Replace with the correct frame if needed
    marker.header.stamp = rospy.Time.now()
    marker.ns = "lines"
    marker.id = 0
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD

    # Line properties
    marker.scale.x = 0.02  # Line width
    marker.color.r = 0.0   # Red
    marker.color.g = 0.0   # Green
    marker.color.b = 1.0   # Blue
    marker.color.a = 1.0   # Alpha (opacity)

    # Parse lines from Float32MultiArray
    data = msg.data
    for i in range(0, len(data), 6):  # Each line is defined by 6 values: x1, y1, z1, x2, y2, z2
        p1 = Point(x=data[i], y=data[i+1], z=data[i+2])
        p2 = Point(x=data[i+3], y=data[i+4], z=data[i+5])
        marker.points.append(p1)
        marker.points.append(p2)

    # Publish the marker
    marker_pub.publish(marker)

if __name__ == '__main__':
    rospy.init_node('line_marker_publisher')

    # Publisher for RViz Markers
    marker_pub = rospy.Publisher('/line_markers', Marker, queue_size=10)

    # Subscriber to the detected lines topic
    rospy.Subscriber('/detected_3d_lines', Float32MultiArray, line_callback)

    rospy.spin()
