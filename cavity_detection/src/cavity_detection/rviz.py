#!/usr/bin/env python3

import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Point
from cavity_detection_msgs.msg import Roi
from cavity_detection.cavity_structs_2 import HorizontalCluster, HorizontalCavity
import tf.transformations
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
import tf
import tf2_ros
from geometry_msgs.msg import Vector3, Quaternion, TransformStamped, PoseStamped, Pose
import tf2_geometry_msgs
import re

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

def draw_roi(roi):
    marker = Marker()
    # Use regex to extract the number from the id
    match = re.search(r'\d+', roi.id)
    if match:
        roi_number = int(match.group())
    else:
        roi_number = 0
    marker.header.frame_id = roi.id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "horiz_clusters"
    marker.id = roi_number
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = roi.length/2
    marker.pose.position.y = roi.width/2
    marker.pose.position.z = roi.height/2
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1
    marker.scale.x = roi.length
    marker.scale.y = roi.width
    marker.scale.z = roi.height
    marker.color.a = 1.0
    marker.color.r = 0
    marker.color.g = 0.5
    marker.color.b = 0.5
    return marker

def draw_cavity(cavity):
    marker = Marker()
    marker.header.frame_id = cavity.parent.id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "horiz_cavities"
    marker.id = cavity.id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = cavity.parent.length/2
    marker.pose.position.y = cavity.width/2
    marker.pose.position.z = cavity.height/2
    marker.pose.orientation.x = 0
    marker.pose.orientation.y = 0
    marker.pose.orientation.z = 0
    marker.pose.orientation.w = 1
    marker.scale.x = cavity.parent.length
    marker.scale.y = cavity.parent.width
    marker.scale.z = cavity.parent.height
    marker.color.a = 1.0
    marker.color.r = 0
    marker.color.g = 0.5
    marker.color.b = 0.5
    return marker

def publish_all(pub, horiz_cavities, vert_cavities):
    # Publish markers for visualization
    array = MarkerArray()
    for roi in horiz_cavities.values():
        marker = draw_roi(roi)
        array.markers.append(marker)
        for cavity in roi.cavities:
            draw_cavity(cavity)
    for roi in vert_cavities.values():
        marker = draw_roi(roi)
        array.markers.append(marker)
        for cavity in roi.cavities:
            draw_cavity(cavity)
    pub.publish(array)

def create_transform(parent, child, translation, rotation):
    transform = TransformStamped()
    transform.header.stamp = rospy.Time.now()
    transform.header.frame_id = parent
    transform.child_frame_id = child
    transform.transform.translation.x = translation[0]
    transform.transform.translation.y = translation[1]
    transform.transform.translation.z = 0
    rotation = R.from_euler('z', rotation).as_quat()
    transform.transform.rotation.x = rotation[0]
    transform.transform.rotation.y = rotation[1]
    transform.transform.rotation.z = rotation[2]
    transform.transform.rotation.w = rotation[3]
    return transform

def publish_transforms(pub, horiz_cavities, vert_cavities):
    transform_list = []
    if len(horiz_cavities) + len(vert_cavities) == 0:
        #print("nada")
        return
    for roi in horiz_cavities.values():
        transform = create_transform("map", roi.id, roi.anchor_point, roi.orientation)
        transform_list.append(transform)
        # print(f"Transform from map to {roi.id}: {roi.anchor_point}, {roi.orientation}")
        if roi.cavities is not None:
            for cavity in roi.cavities:
                transform = create_transform(roi.id, cavity.id, cavity.front, roi.orientation)
                transform_list.append(transform)

    pub.sendTransform(transform_list)
    print(f"Published {len(transform_list)} transforms")


def vert_detector_markers(i, points, vertices, x1, y1, x2, y2):
    """Publishes a marker containing both the triangle and highlighted cells."""

    # Triangle edges (LINE_STRIP)
    triangle_marker = Marker()
    triangle_marker.header.frame_id = "map"
    triangle_marker.header.stamp = rospy.Time.now()
    triangle_marker.ns = "triangle"
    triangle_marker.id = i
    triangle_marker.type = Marker.LINE_STRIP
    triangle_marker.action = Marker.ADD
    triangle_marker.scale.x = 0.05  # Line thickness
    triangle_marker.color.r = 0.0
    triangle_marker.color.g = 1.0
    triangle_marker.color.b = 0.0
    triangle_marker.color.a = 1.0

    for v in vertices:
        p = Point()
        p.x, p.y, _ = v
        p.z = 0
        triangle_marker.points.append(p)

    # Close the triangle
    triangle_marker.points.append(triangle_marker.points[0])

    # Highlighted occupied cells (POINTS)
    points_marker = Marker()
    points_marker.header.frame_id = "map"
    points_marker.header.stamp = rospy.Time.now()
    points_marker.ns = "highlighted_pts"
    points_marker.id = i
    points_marker.type = Marker.POINTS
    points_marker.action = Marker.ADD
    points_marker.scale.x = 0.1  # Adjust for grid resolution
    points_marker.scale.y = 0.1
    points_marker.color.r = 1.0
    points_marker.color.g = 0.0
    points_marker.color.b = 0.0
    points_marker.color.a = 1.0

    for pt in points:
        p = Point()
        p.x, p.y = pt
        p.z = 0
        points_marker.points.append(p)
    
    # create markers for p1 and p2
    p1_marker = Marker()
    p1_marker.header.frame_id = "map"
    p1_marker.header.stamp = rospy.Time.now()
    p1_marker.ns = "p1"
    p1_marker.id = i
    p1_marker.type = Marker.SPHERE
    p1_marker.action = Marker.ADD
    p1_marker.scale.x = 0.1
    p1_marker.scale.y = 0.1
    p1_marker.scale.z = 0.1
    p1_marker.color.r = 0.0
    p1_marker.color.g = 0.0
    p1_marker.color.b = 1.0
    p1_marker.color.a = 1.0
    p1_marker.pose.position.x = x1
    p1_marker.pose.position.y = y1
    p1_marker.pose.position.z = 0
    p1_marker.pose.orientation.w = 1.0
    p1_marker.pose.orientation.x = 0.0
    p1_marker.pose.orientation.y = 0.0
    p1_marker.pose.orientation.z = 0.0
    p2_marker = Marker()
    p2_marker.header.frame_id = "map"
    p2_marker.header.stamp = rospy.Time.now()
    p2_marker.ns = "p2"
    p2_marker.id = i
    p2_marker.type = Marker.SPHERE
    p2_marker.action = Marker.ADD
    p2_marker.scale.x = 0.1
    p2_marker.scale.y = 0.1
    p2_marker.scale.z = 0.1 
    p2_marker.color.r = 0.0
    p2_marker.color.g = 0.0
    p2_marker.color.b = 1.0
    p2_marker.color.a = 1.0
    p2_marker.pose.position.x = x2
    p2_marker.pose.position.y = y2
    p2_marker.pose.position.z = 0
    p2_marker.pose.orientation.w = 1.0
    p2_marker.pose.orientation.x = 0.0
    p2_marker.pose.orientation.y = 0.0
    p2_marker.pose.orientation.z = 0.0

    # Publish both markers
    return triangle_marker, points_marker, p1_marker, p2_marker