#!/usr/bin/env python3

import rospy
import numpy as np
from cavity_detection_msgs.msg import Roi
import tf.transformations
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
import tf
import tf2_ros
from geometry_msgs.msg import Vector3, Quaternion, TransformStamped, PoseStamped, Pose
import tf2_geometry_msgs

def publish_temporal(pub, msg):
    marker = Marker()
    marker.header = msg.header
    marker.ns = "temporal"
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = msg.center.x
    marker.pose.position.y = msg.center.y
    marker.pose.position.z = msg.depth / 2.0
    marker.pose.orientation = msg.orientation
    marker.scale.x = msg.length
    marker.scale.y = msg.width
    marker.scale.z = msg.depth
    marker.color.a = 0.5
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    if msg.roi_type == 0:
        marker.pose.position.z = -msg.depth / 2.0
        marker.scale.x = msg.length
        marker.scale.y = msg.width
        marker.scale.z = msg.depth
    elif msg.roi_type == 1:
        marker.scale.x = 0.05
        marker.scale.y = msg.width
        marker.scale.z = msg.depth

    pub.publish(marker)

def publish_all(pub, horiz_cavities):
    # Publish markers for visualization
    array = MarkerArray()
    for i, roi in enumerate(horiz_cavities):
        marker = Marker()
        marker.header.frame_id = roi.id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "horiz_cavities"
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = roi.length/2
        marker.pose.position.y = roi.width/2
        marker.pose.position.z = roi.depth
        marker.pose.orientation.x = roi.orientation[0]
        marker.pose.orientation.y = roi.orientation[1]
        marker.pose.orientation.z = roi.orientation[2]
        marker.pose.orientation.w = roi.orientation[3]
        marker.scale.x = roi.length
        marker.scale.y = roi.width
        marker.scale.z = roi.depth
        marker.color.a = 1.0
        marker.color.r = 0
        marker.color.g = 0.5
        marker.color.b = 0.5
        array.markers.append(marker)
    pub.publish(array)