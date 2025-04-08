#!/usr/bin/env python3

import rospy
import numpy as np
from cavity_detection_msgs.msg import Roi, RoiStamped, HorizontalObservation, VerticalObservation
from cavity_detection_msgs.srv import UpdateRoi, UpdateRoiResponse, GetNearestRoi, GetNearestRoiResponse, AddCavity, AddCavityResponse, UpdateCavity, UpdateCavityResponse
import tf.transformations
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
import tf
import tf2_ros
from geometry_msgs.msg import Vector3, Quaternion, TransformStamped, PoseStamped, Pose, Point
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion
from tf2_geometry_msgs import do_transform_pose
from scipy.spatial.transform import Rotation
from cavity_detection.cavity_structs_2 import HorizontalCluster, HorizontalCavity, VerticalCluster, VerticalCavity
from cavity_detection.rviz import publish_temporal, publish_all, publish_transforms
from cavity_detection.helpers import transform_2d, invert_2d_transform
from scipy.spatial import KDTree

verbose = False
WALL_DEPTH = 0.2
TF_RATE = 1


class CavityMap:
    def __init__(self):
        self.horiz_clusters = {}
        self.vert_clusters = {}
        rospy.init_node('cavity_map', anonymous=True)
        rospy.Subscriber('/horiz_roi', HorizontalObservation, self.horiz_callback)
        # rospy.Subscriber('/vert_roi', VerticalObservation, self.vert_callback)
        # Subscribe to the tf of the camera in map frame
        self.tf_listener = tf.TransformListener()
        self.tf_pub = tf2_ros.TransformBroadcaster()
        self.marker_pub = rospy.Publisher('/rois', MarkerArray, queue_size=2)
        self.s1 = rospy.Service('get_nearest_roi', GetNearestRoi, self.handle_get_nearest_roi)
        self.s2 = rospy.Service('update_roi', UpdateRoi, self.handle_update_roi)
        self.s3 = rospy.Service('add_cavity', AddCavity, self.handle_add_cavity)
        self.s4 = rospy.Service('update_cavity', UpdateCavity, self.handle_update_cavity)
        
        self.kd_tree = None
        self.open_cavities = []

    def horiz_callback(self, msg):
        updated = None
        observation_angle = msg.orientation
        # observation_distance = np.linalg.norm(np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]))
        try:
            pos, quat = self.tf_listener.lookupTransform("map", msg.header.frame_id, msg.header.stamp)
            angle = euler_from_quaternion(quat)[2]
            T = np.array([[np.cos(angle),-np.sin(angle), pos[0]], 
                          [np.sin(angle),np.cos(angle), pos[1]], 
                          [0, 0, 1]])
            transformed_msg = self.transform_observation(msg, T)
            for roi in self.horiz_clusters.values():
                if roi.is_overlapping(transformed_msg):
                    roi.add_observation(transformed_msg)
                    updated = True
                    if verbose: print(f"Updated cavity {roi.id}")
                    break
            if not updated:
                roi_id = f'horiz_roi_{len(self.horiz_clusters)}'
                new_roi = HorizontalCluster(roi_id, transformed_msg)
                self.horiz_clusters[roi_id] = new_roi
                if verbose: print(f"Added new cavity {roi_id}")
            
            self.make_tree()

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"Transform error: {e}")

    def vert_callback(self, msg):
        updated = False
        observation_angle = msg.orientation
        # observation_distance = np.linalg.norm(np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]))
        try:
            pos, quat = self.tf_listener.lookupTransform("map", msg.header.frame_id, msg.header.stamp)
            
            transform_stamped = TransformStamped()
            transform_stamped.transform.translation = Vector3(pos[0], pos[1], pos[2])
            transform_stamped.transform.rotation = Quaternion(quat[0], quat[1], quat[2], quat[3])
            transform_stamped.child_frame_id = 'map'
            transform_stamped.header = msg.header
            transformed_msg = self.transform_roi(msg, transform_stamped)
            for i, roi in enumerate(self.vert_clusters.values()):
                if roi.is_overlapping(transformed_msg):
                    roi.add_observation(transformed_msg)
                    updated = True
                    if verbose: print(f"Updated cavity {roi.id}")
                    break
            if not updated:
                roi_id = f'vert_roi_{len(self.vert_clusters)}'
                new_roi = VerticalCluster(roi_id, transformed_msg)
                self.vert_clusters[roi_id] = new_roi
                if verbose: print(f"Added new cavity {roi_id}")

            self.make_tree()

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"Transform error: {e}")
    
    def merge_clusters(self, cluster_1_id, cluster_2_id):
        if verbose: print(f"Merging clusters {cluster_1_id} and {cluster_2_id}")
        # find the bigger cluster
        # make smaller one into an observation message
        # do fusion
        # remove smaller cluster from set
        self.horiz_clusters.__delitem__(cluster_2_id)
    
    def make_tree(self):
        self.open_cavities = []
        for roi in self.horiz_clusters.values():
            self.open_cavities.append(roi)
        self.kd_tree = KDTree([roi.anchor_point for roi in self.open_cavities])

    def transform_observation(self, observation, transform):
        lines = np.array(observation.lines).reshape(-1, 2)
        lines = np.vstack((lines.T, np.ones((1, len(lines))))) 
        transformed_lines = np.dot(transform, lines)
        transformed_lines = transformed_lines[:2, :].T
        transformed_lines = transformed_lines.flatten()
        observation.lines = transformed_lines.tolist()
        observation.header.frame_id = "map"
        return observation

    def handle_get_nearest_roi(self, req):
        # Logic to handle the request and generate a response
        robot_pos, _ = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
        if self.kd_tree is None:
            self.make_tree()
            if verbose: print("made tree")
        robot_xy = [robot_pos[0], robot_pos[1]]
        closest_index = self.kd_tree.query(robot_xy)[1]
        closest_roi = self.open_cavities[closest_index]
        response = GetNearestRoiResponse()
        response.roi = Roi()
        response.roi.pose.position = Point(x=closest_roi.origin[0], y=closest_roi.origin[1], z=0)
        if verbose: print(f"Closest ROI position: {response.roi.pose.position}")
        response.roi.pose.orientation = Quaternion(x=closest_roi.orientation[0], y=closest_roi.orientation[1], z=closest_roi.orientation[2], w=closest_roi.orientation[3])
        response.roi.length = closest_roi.length
        response.roi.width = closest_roi.width
        response.roi.depth = closest_roi.depth
        response.roi.num_cavities = closest_roi.num_cavities
        response.roi.cavity_width = closest_roi.cavity_width
        return response

    def handle_update_roi(self, req):
        # Logic to update status and generate response
        roi_id = req.roi_id
        roi = self.horiz_cavities[roi_id]
        if req.roi_pose.position.x != 0 or req.roi_pose.position.y != 0:
            roi.position = np.array([req.roi_pose.position.x, req.roi_pose.position.y, req.roi_pose.position.z])
            roi.orientation = np.array([req.roi_pose.orientation.x, req.roi_pose.orientation.y, req.roi_pose.orientation.z, req.roi_pose.orientation.w])
        if req.length != 0.0:
            roi.length = req.length
        if req.width != 0.0:
            roi.width = req.width
        if req.depth != 0.0:
            roi.depth = req.depth
        response = UpdateRoiResponse()
        response.success = True
        return response

    def handle_add_cavity(self, req):
        # Logic to update status and generate response
        roi_id = req.roi_id
        roi = self.horiz_cavities[roi_id]
        next_cavity = len(roi.cavities)
        cavity_id = f'{roi_id}_{next_cavity}'
        front = np.array([0, req.y_offset, 0])
        width = req.width
        cavity = HorizontalCavity(cavity_id, roi_id, front, roi.orientation, roi.length, width, roi.depth)

        roi.cavities[cavity_id] = cavity
        response = AddCavityResponse()
        response.cavity_id = cavity_id
        response.success = True
        return response

    def handle_update_cavity(self, req):
        # Logic to update status and generate response
        roi_id = req.roi_id
        cavity_id = req.cavity_id
        roi = self.horiz_cavities[roi_id]
        cavity = roi.cavities[cavity_id]
        if req.y_offset != 0.0:
            roi.y_offset = req.y_offset
        if req.width != 0.0:
            cavity.width = req.width
        if req.status:
            cavity.status = req.status
        response = UpdateCavityResponse()
        response.success = True
        return response
    
    def run_publish_markers(self, event=None):
        publish_transforms(self.tf_pub, self.horiz_clusters, self.vert_clusters)

        publish_all(self.marker_pub, self.horiz_clusters, self.vert_clusters)
    
    def run(self):
        # Set up a timer to publish markers at a fixed rate
        rospy.Timer(rospy.Duration(1.0/TF_RATE), self.run_publish_markers)
        if verbose: print("Timer set")
        rospy.spin()

def test_callback(event):
    # This function is just for testing purposes
    # It will be called every second
    if verbose: print("Test callback triggered")
    
def run():        
    cavity_map = CavityMap()
    cavity_map.run()

if __name__ == '__main__':
    try:
        run()
    except rospy.ROSInterruptException:
        pass
