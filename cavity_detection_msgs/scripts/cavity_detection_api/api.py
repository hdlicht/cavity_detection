#!/usr/bin/env python3

import rospy
from cavity_detection_msgs.srv import * # AddRoi, GetNearestRoi, GetRoiById, MoveRoi, UpdateRoi, AddCavity, UpdateCavity, MarkFilled, MarkTarget
from geometry_msgs.msg import Pose, Point, Quaternion
from cavity_detection_msgs.msg import Roi

def add_roi(x, y, theta, length, height, spacing, num_cavities):
    rospy.wait_for_service('add_roi')
    try:
        add_roi_proxy = rospy.ServiceProxy('add_roi', AddRoi)
        req = AddRoiRequest()
        req.x = x
        req.y = y
        req.theta = theta
        req.length = length
        req.height = height
        req.spacing = spacing
        req.num_cavities = num_cavities
        resp = add_roi_proxy(req)
        rospy.loginfo(f"Created ROI: {resp.roi_id}")
        return resp.roi_id
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        
def get_nearest_roi():
    print("Waiting for get_nearest_roi service...")
    rospy.wait_for_service('get_nearest_roi')
    print("Done waiting!")
    try:
        get_nearest_roi_proxy = rospy.ServiceProxy('get_nearest_roi', GetNearestRoi)
        req = GetNearestRoi()
        resp = get_nearest_roi_proxy()

        rospy.loginfo(f"Nearest cavity ID: {resp.roi.id}")
        return resp.roi
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def get_roi_by_id(roi_id):
    print("Waiting for get_roi_by_id service...")
    rospy.wait_for_service('get_roi_by_id')
    print("Done waiting!")
    try:
        get_roi_by_id_proxy = rospy.ServiceProxy('get_roi_by_id', GetRoiById)
        req = GetRoiByIdRequest()
        req.id = roi_id
        resp = get_roi_by_id_proxy()
        rospy.loginfo(f"Retrieving {req.id}")
        return resp.roi
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def move_roi(roi_id, dtheta, dx, dy):
    rospy.wait_for_service('move_roi')
    try:
        move_roi_proxy = rospy.ServiceProxy('move_roi', MoveRoi)
        req = MoveRoiRequest()
        req.roi_id = roi_id
        req.dtheta = dtheta
        req.dx = dx
        req.dy = dy
        resp = move_roi_proxy(req)
        rospy.loginfo(f"Move ROI success: {resp.success}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def update_roi(roi_id, length, height, spacing, num_cavities):
    rospy.wait_for_service('update_roi')
    try:
        update_roi_proxy = rospy.ServiceProxy('update_roi', UpdateRoi)
        req = UpdateRoiRequest()
        req.roi_id = roi_id
        req.length = length
        req.height = height
        req.spacing = spacing
        req.num_cavities = num_cavities
        resp = update_roi_proxy(req)
        rospy.loginfo(f"Update ROI success: {resp.success}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def split_roi(roi_id, start, end):
    rospy.wait_for_service('update_roi')
    try:
        split_roi_proxy = rospy.ServiceProxy('split_roi', SplitRoi)
        req = SplitRoiRequest()
        req.roi_id = roi_id
        req.start = start
        req.end = end
        resp = split_roi_proxy(req)
        rospy.loginfo(f"Split ROI success: {resp.success}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def mark_filled(roi_id):
    rospy.wait_for_service('mark_filled')
    try:
        mark_filled_proxy = rospy.ServiceProxy('mark_filled', MarkFilled)
        req = MarkFilledRequest()
        req.roi_id = roi_id
        resp = mark_filled_proxy(req)
        rospy.loginfo(f"Mark Filled success: {resp.success}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def mark_target(roi_id):
    rospy.wait_for_service('mark_target')
    try:
        mark_target_proxy = rospy.ServiceProxy('mark_target', MarkTarget)
        req = MarkTargetRequest()
        req.roi_id = roi_id
        resp = mark_target_proxy(req)
        rospy.loginfo(f"Mark Target success: {resp.success}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def add_cavity(roi_id, y_offset, width):
    rospy.wait_for_service('add_cavity')
    try:
        add_cavity_proxy = rospy.ServiceProxy('add_cavity', AddCavity)
        req = AddCavityRequest()
        req.roi_id = roi_id
        req.y_offset = y_offset
        req.width = width
        resp = add_cavity_proxy(req)
        rospy.loginfo(f"Added cavity ID: {resp.cavity_id}, Success: {resp.success}")
        return resp.cavity_id
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

def update_cavity(roi_id, cavity_id, y_offset, width, status):
    rospy.wait_for_service('update_cavity')
    try:
        update_cavity_proxy = rospy.ServiceProxy('update_cavity', UpdateCavity)
        req = UpdateCavityRequest()
        req.roi_id = roi_id
        req.cavity_id = cavity_id
        req.y_offset = y_offset
        req.width = width
        req.status = status
        resp = update_cavity_proxy(req)
        rospy.loginfo(f"Update cavity success: {resp.success}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")