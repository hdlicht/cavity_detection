#!/usr/bin/env python3

import rospy
import numpy as np
from cavity_detection_msgs.msg import Roi
from cavity_detection_msgs.srv import UpdateRoi, UpdateRoiResponse, GetNearestCavity, GetNearestCavityResponse, AddCavity, AddCavityResponse, UpdateCavity, UpdateCavityResponse
import tf.transformations
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
import tf
import tf2_ros
from geometry_msgs.msg import Vector3, Quaternion, TransformStamped, PoseStamped, Pose
import tf2_geometry_msgs
from tf2_geometry_msgs import do_transform_pose
from scipy.spatial.transform import Rotation
from cavity_detection.cavity_structs import Cavity, Observation, RegionOfInterest
from cavity_detection.rviz import publish_temporal, publish_all
from scipy.spatial import KDTree


WALL_DEPTH = 0.2
TF_RATE = 1


class CavityMap:
    def __init__(self):
        self.horiz_cavities = {}
        self.vert_cavities = {}
        rospy.init_node('cavity_map', anonymous=True)
        rospy.Subscriber('/horiz_roi', Roi, self.horiz_callback)
        rospy.Subscriber('/vert_roi', Roi, self.vert_callback)
        # Subscribe to the tf of the camera in map frame
        self.tf_listener = tf.TransformListener()
        self.tf_pub = tf2_ros.TransformBroadcaster()
        self.marker_pub = rospy.Publisher('/rois', MarkerArray, queue_size=2)
        self.s1 = rospy.Service('get_nearest_cavity', GetNearestCavity, self.handle_get_nearest_cavity)
        self.s2 = rospy.Service('update_roi', UpdateRoi, self.handle_update_roi)
        self.s3 = rospy.Service('add_cavity', AddCavity, self.handle_add_cavity)
        self.s4 = rospy.Service('update_cavity', UpdateCavity, self.handle_update_cavity)
        self.kd_tree = None
        self.open_cavities = []

    def horiz_callback(self, msg):
        print("got one")
        updated = False
        observation_angle = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        observation_distance = np.linalg.norm(np.array([msg.origin.x, msg.origin.y, msg.origin.z]))
        # publish_temporal(msg)
        try:
            pos, quat = self.tf_listener.lookupTransform("map", msg.header.frame_id, rospy.Time(0))
            transform_stamped = TransformStamped()
            transform_stamped.transform.translation = Vector3(pos[0], pos[1], pos[2])
            transform_stamped.transform.rotation = Quaternion(quat[0], quat[1], quat[2], quat[3])
            transform_stamped.child_frame_id = 'map'
            transform_stamped.header = msg.header
            transformed_msg = self.transform_roi(msg, "map", transform_stamped)
            observation = Observation(transformed_msg.origin, transformed_msg.orientation, 
                                    transformed_msg.length, transformed_msg.width, 
                                    transformed_msg.depth, observation_distance, observation_angle, transformed_msg.num_cavities, transformed_msg.cavity_width)
            for i, roi in enumerate(self.horiz_cavities.values()):
                if observation.is_overlapping(roi):
                    roi.add_observation(observation)
                    updated = True
                    #print(f"Updated cavity {roi.id}")
                    break
            if not updated:
                roi_id = f'roi_{len(self.horiz_cavities)}'
                new_roi = RegionOfInterest(roi_id, 0, observation)
                self.horiz_cavities[roi_id] = new_roi
                #print(f"Added new cavity {roi_id}")
            
            self.make_tree()

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"Transform error: {e}")

    def vert_callback(self, msg):
        updated = False
        observation_angle = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        observation_distance = np.linalg.norm(np.array([msg.origin.x, msg.origin.y, msg.origin.z]))
        # publish_temporal(msg)
        try:
            pos, quat = self.tf_listener.lookupTransform("map", msg.header.frame_id, rospy.Time(0))
            
            transform_stamped = TransformStamped()
            transform_stamped.transform.translation = Vector3(pos[0], pos[1], pos[2])
            transform_stamped.transform.rotation = Quaternion(quat[0], quat[1], quat[2], quat[3])
            transform_stamped.child_frame_id = 'map'
            transform_stamped.header = msg.header
            transformed_msg = self.transform_roi(msg, "map", transform_stamped)
            observation = Observation(transformed_msg.origin, transformed_msg.orientation, WALL_DEPTH, transformed_msg.width, transformed_msg.depth, observation_distance, observation_angle)
            for i, roi in enumerate(self.horiz_cavities):
                if observation.is_overlapping(roi):
                    roi.add_observation(observation)
                    updated = True
                    #print(f"Updated cavity {roi.id}")
                    break
            if not updated:
                roi_id = f'roi_{len(self.vert_cavities)}'
                new_roi = RegionOfInterest(roi_id, 1, observation)
                self.horiz_cavities[roi_id] = new_roi
                #print(f"Added new cavity {roi_id}")

            self.make_tree()

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"Transform error: {e}")

    def create_transform(self, parent, child, translation, rotation):
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = parent
        transform.child_frame_id = child
        transform.transform.translation.x = translation[0]
        transform.transform.translation.y = translation[1]
        transform.transform.translation.z = translation[2]
        transform.transform.rotation.x = rotation[0]
        transform.transform.rotation.y = rotation[1]
        transform.transform.rotation.z = rotation[2]
        transform.transform.rotation.w = rotation[3]
        return transform
    
    def publish_transforms(self, event=None):
        transform_list = []
        if len(self.horiz_cavities) + len(self.vert_cavities) == 0:
            #print("nada")
            return
        for roi in self.horiz_cavities.values():
            transform = self.create_transform("map", roi.id, roi.origin, roi.orientation)
            transform_list.append(transform)
            if roi.cavities is not None:
                for cavity in roi.cavities:
                    transform = self.create_transform(roi.id, cavity.id, cavity.front, roi.orientation)
                    transform_list.append(transform)

        self.tf_pub.sendTransform(transform_list)

        publish_all(self.marker_pub, self.horiz_cavities)

        
    
    def make_tree(self):
        self.open_cavities = []
        for roi in self.horiz_cavities.values():
            self.open_cavities.append(roi)
        self.kd_tree = KDTree([roi.origin[:2] for cavity in self.open_cavities])

    def transform_roi(self, roi, target_frame, transform):
        
        # Create a PoseStamped for the orientation.
        # The position here is arbitrary since we only care about the orientation.
        pose = PoseStamped()
        pose.pose.position = roi.origin
        pose.pose.orientation = roi.orientation
        pose.header = roi.header
        pose.header.stamp = rospy.Time(0)
        
        # Transform the pose
        transformed_pose = do_transform_pose(pose, transform)
        
        # Update the roi with the transformed data
        roi.origin = transformed_pose.pose.position
        roi.orientation = transformed_pose.pose.orientation
        roi.header.frame_id = target_frame
        return roi

    # def publish_markers(self, event):
    #     publish_all(self.pub, self.horiz_cavities, self.vert_cavities)

    def handle_get_nearest_cavity(self, req):
        print("entered handler")
        # Logic to handle the request and generate a response
        if self.kd_tree is None:
            self.make_tree()
            print("made tree")
        robot_xy = [req.robot_pose.position.x, req.robot_pose.position.y]
        closest_index = self.kd_tree.query(robot_xy)[1]
        cavity = self.open_cavities[closest_index]
        response = GetNearestCavityResponse()
        response.cavity_pose = Pose()
        response.cavity_pose.position.x = cavity.origin[0]
        response.cavity_pose.position.y = cavity.origin[1]
        response.cavity_pose.position.z = cavity.depth
        response.cavity_pose.orientation.x = cavity.orientation[0]
        response.cavity_pose.orientation.y = cavity.orientation[1]
        response.cavity_pose.orientation.z = cavity.orientation[2]
        response.cavity_pose.orientation.w = cavity.orientation[3]
        response.cavity_id = cavity.id
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
        parent = roi_id
        front = np.array([0, req.y_offset, 0])
        width = req.width
        cavity = Cavity(cavity_id, roi_id, front, roi.orientation, roi.length, width, roi.depth)

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

    def run(self):
        rospy.Timer(rospy.Duration(1.0), self.publish_transforms, oneshot=False)
        rospy.spin()


if __name__ == '__main__':
    try:
        cavity_map = CavityMap()
        cavity_map.run()
    except rospy.ROSInterruptException:
        pass
