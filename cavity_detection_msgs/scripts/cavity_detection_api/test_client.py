#!/usr/bin/env python3

import rospy
from cavity_detection_msgs.srv import *
from cavity_detection_api.api import get_nearest_roi, add_roi, update_roi, add_cavity, update_cavity
from geometry_msgs.msg import Pose, Point, Quaternion

def main():
    rospy.init_node('cavity_client_node')

    x = -2.0
    y = -1.5
    theta = 0.0
    length = 2.2
    height = 0.22
    spacing = 0.36
    num_cavities = 6

    roi_id = add_roi(x, y, theta, length, height, spacing, num_cavities)
    print(roi_id)
    # # Test get_nearest_cavity service
    # roi_id, roi_pose = get_nearest_roi()

    # # Example ROI pose for testing
    # new_roi_pose = Pose()
    # new_roi_pose.position.x = 1.0
    # new_roi_pose.position.y = 2.0
    # new_roi_pose.position.z = 0.0
    # new_roi_pose.orientation.x = 0.0
    # new_roi_pose.orientation.y = 0.0
    # new_roi_pose.orientation.z = 0.0
    # new_roi_pose.orientation.w = 1.0

    # # Test update_roi service
    # update_roi(roi_id=roi_id, roi_pose=roi_pose, length=1.0, width=1.0, depth=1.0)

    # # Test add_cavity service
    # cavity_id = add_cavity(roi_id=roi_id, y_offset=0.5, width=0.2)
    # print(cavity_id)
    # # Test update_cavity service
    # update_cavity(roi_id=roi_id, cavity_id=cavity_id, y_offset=3.5, width=0.5, status=True)

if __name__ == '__main__':
    main()