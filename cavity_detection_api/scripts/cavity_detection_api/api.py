#!/usr/bin/env python3

import rospy
from cavity_detection_msgs.srv import GetNearestRoi, GetRoiById, MoveRoi, UpdateRoi, AddCavity, UpdateCavity
from geometry_msgs.msg import Pose, Point, Quaternion
from cavity_detection_msgs.msg import Roi, RoiStamped

def get_nearest_roi():
    print("Waiting for get_nearest_roi service...")
    rospy.wait_for_service('get_nearest_roi')
    print("Done waiting!")
    try:
        get_nearest_roi = rospy.ServiceProxy('get_nearest_roi', GetNearestRoi)
        req = GetNearestRoi()
        resp = get_nearest_roi()

        rospy.loginfo(f"Nearest cavity ID: {resp.roi.id}")
        return resp.roi
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def get_roi_by_id():
    print("Waiting for get_roi_by_id service...")
    rospy.wait_for_service('get_roi_by_id')
    print("Done waiting!")
    try:
        get_roi_by_id = rospy.ServiceProxy('get_roi_by_id', GetNearestRoi)
        req = GetRoiById()
        resp = get_roi_by_id()

        rospy.loginfo(f"Retrieving {resp.roi.id}")
        return resp.roi
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def move_roi(roi_id, dtheta, dx, dy):
    rospy.wait_for_service('move_roi')
    try:
        update_roi = rospy.ServiceProxy('move_roi', UpdateRoi)
        req = UpdateRoi()
        req.roi_id = roi_id
        req.dtheta = dtheta
        req.dx = dx
        req.dy = dy
        resp = update_roi(req)
        rospy.loginfo(f"Move ROI success: {resp.success}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def update_roi(roi_id, length, height, spacing, num_cavities):
    rospy.wait_for_service('update_roi')
    try:
        update_roi = rospy.ServiceProxy('update_roi', UpdateRoi)
        req = UpdateRoi()
        req.roi_id = roi_id
        req.length = length
        req.height = height
        req.depth = spacing
        req.num_cavities = num_cavities
        resp = update_roi(req)
        rospy.loginfo(f"Update ROI success: {resp.success}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def add_cavity(roi_id, y_offset, width):
    rospy.wait_for_service('add_cavity')
    try:
        add_cavity = rospy.ServiceProxy('add_cavity', AddCavity)
        req = AddCavity()
        req.roi_id = roi_id
        req.y_offset = y_offset
        req.width = width
        resp = add_cavity(req)
        rospy.loginfo(f"Added cavity ID: {resp.cavity_id}, Success: {resp.success}")
        return resp.cavity_id
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def update_cavity(roi_id, cavity_id, y_offset, width, status):
    rospy.wait_for_service('update_cavity')
    try:
        update_cavity = rospy.ServiceProxy('update_cavity', UpdateCavity)
        req = UpdateCavity()
        req.roi_id = roi_id
        req.cavity_id = cavity_id
        req.y_offset = y_offset
        req.width = width
        req.status = status
        resp = update_cavity(req)
        rospy.loginfo(f"Update cavity success: {resp.success}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")