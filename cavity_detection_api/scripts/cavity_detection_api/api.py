#!/usr/bin/env python3

import rospy
from cavity_detection_msgs.srv import GetNearestRoi, UpdateRoi, UpdateRoiRequest, AddCavity, AddCavityRequest, UpdateCavity, UpdateCavityRequest
from geometry_msgs.msg import Pose, Point, Quaternion
from cavity_detection_msgs.msg import Roi, RoiStamped

def get_nearest_cavity():
    rospy.wait_for_service('get_nearest_cavity')
    print("done waiting")
    try:
        get_nearest_roi = rospy.ServiceProxy('get_nearest_roi', GetNearestRoi)
        req = GetNearestRoi()
        resp = get_nearest_roi(req)

        rospy.loginfo(f"Nearest cavity ID: {resp.roi.id}, Pose: {resp.roi.pose}")
        return resp.roi
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def update_roi(roi_id, roi_pose, length, width, depth):
    rospy.wait_for_service('update_roi')
    try:
        update_roi = rospy.ServiceProxy('update_roi', UpdateRoi)
        req = UpdateRoiRequest()
        req.roi_id = roi_id
        req.roi_pose = roi_pose
        req.length = length
        req.width = width
        req.depth = depth
        resp = update_roi(req)
        rospy.loginfo(f"Update ROI success: {resp.success}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def add_cavity(roi_id, y_offset, width):
    rospy.wait_for_service('add_cavity')
    try:
        add_cavity = rospy.ServiceProxy('add_cavity', AddCavity)
        req = AddCavityRequest()
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
        req = UpdateCavityRequest()
        req.roi_id = roi_id
        req.cavity_id = cavity_id
        req.y_offset = y_offset
        req.width = width
        req.status = status
        resp = update_cavity(req)
        rospy.loginfo(f"Update cavity success: {resp.success}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")