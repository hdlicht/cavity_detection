#!/usr/bin/env python3
 
import rospy
import numpy as np
from cavity_detection.msg import Roi
import tf.transformations
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
import tf
import tf2_ros
from geometry_msgs.msg import Vector3, Quaternion, TransformStamped, Pose
import tf2_geometry_msgs
from tf2_geometry_msgs import do_transform_pose


WALL_DEPTH = 0.2

class RegionOfInterest:    
    def __init__(self, id, roi_type, first_observation):
        self.type = roi_type
        self.id = id
        self.center = first_observation.center
        self.orientation = first_observation.orientation
        self.length = first_observation.length
        self.width = first_observation.width
        self.depth = first_observation.depth
        self.observations = []
        self.observations.append(first_observation)

    def add_observation(self, observation):
        count = len(self.observations)
        self.observations.append(observation)
        if count > 20:
            self.observations.pop(0)
        self.center = (self.center * count + observation.center) / (count + 1)
        self.orientation = (self.orientation * count + observation.orientation) / (count + 1)
        self.length = (self.length * count + observation.length) / (count + 1)
        self.width = (self.width * count + observation.width) / (count + 1)
        self.depth = (self.depth * count + observation.depth) / (count + 1)


class Observation:
    def __init__(self, center, orientation, length, width, depth, observation_distance, observation_angle):
        self.center = np.array([center.x, center.y, center.z])
        self.orientation = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
        self.length = length
        self.width = width
        self.depth = depth
        self.observation_distance = observation_distance
        self.observation_angle = observation_angle

class CavityMap:
    def __init__(self):
        self.horiz_cavities = []
        self.vert_cavities = []
        rospy.init_node('cavity_map', anonymous=True)
        rospy.Subscriber('/horiz_roi', Roi, self.horiz_callback)
        rospy.Subscriber('/vert_roi', Roi, self.vert_callback)
        # Subscribe to the tf of the camera in map frame
        self.tf_listener = tf.TransformListener()
        self.pub = rospy.Publisher('/markers', MarkerArray, queue_size=2)
        self.pub2 = rospy.Publisher('/temporal', Marker, queue_size=2)

    def publish_temporal(self, msg):
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

        self.pub2.publish(marker)

    def horiz_callback(self, msg):
        updated = False
        observation_angle = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        observation_distance = np.linalg.norm(np.array([msg.center.x, msg.center.y, msg.center.z]))
        self.publish_temporal(msg)
        try:
            pos, quat = self.tf_listener.lookupTransform("map", msg.header.frame_id, rospy.Time(0))
            
            transform_stamped = TransformStamped()
            transform_stamped.transform.translation = Vector3(pos)
            transform_stamped.transform.rotation = Quaternion(quat)
            transform_stamped.child_frame_id = 'map'
            transform_stamped.header = msg.header


            transformed_msg = self.transform_roi(msg, "map", transform_stamped)
            observation = Observation(transformed_msg.center, transformed_msg.orientation, 
                                    transformed_msg.length, transformed_msg.width, 
                                    transformed_msg.depth, observation_distance, observation_angle)
            for i, roi in enumerate(self.horiz_cavities):
                if self.is_overlapping(roi, observation):
                    roi.add_observation(observation)
                    updated = True
                    print(f"Updated cavity {roi.id}")
                    break
            if not updated:
                roi_id = len(self.horiz_cavities) + len(self.vert_cavities)
                new_roi = RegionOfInterest(roi_id, 0, observation)
                self.horiz_cavities.append(new_roi)
                print(f"Added new cavity {roi_id}")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"Transform error: {e}")


    def vert_callback(self, msg):
        updated = False
        observation_angle = np.array([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        observation_distance = np.linalg.norm(np.array([msg.center.x, msg.center.y, msg.center.z]))
        self.publish_temporal(msg)
        try:
            pos, quat = self.tf_listener.lookupTransform("map", msg.header.frame_id, rospy.Time(0))
            
            transform_stamped = TransformStamped()
            transform_stamped.transform.translation = Vector3(pos[0], pos[1], pos[2])
            transform_stamped.transform.rotation = Quaternion(quat[0], quat[1], quat[2], quat[3])
            transform_stamped.child_frame_id = 'map'
            transform_stamped.header = msg.header


            transformed_msg = self.transform_roi(msg, "map", transform_stamped)
            observation = Observation(transformed_msg.center, transformed_msg.orientation, WALL_DEPTH, transformed_msg.width, transformed_msg.depth, observation_distance, observation_angle)
            for i, roi in enumerate(self.horiz_cavities):
                if self.is_overlapping(roi, observation):
                    roi.add_observation(observation)
                    updated = True
                    print(f"Updated cavity {roi.id}")
                    break
            if not updated:
                roi_id = len(self.horiz_cavities) + len(self.vert_cavities)
                new_roi = RegionOfInterest(roi_id, 1, observation)
                self.horiz_cavities.append(new_roi)
                print(f"Added new cavity {roi_id}")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"Transform error: {e}")

    def transform_roi(self, roi, target_frame, transform):
        
        # Create a PoseStamped for the orientation.
        # The position here is arbitrary since we only care about the orientation.
        pose = Pose()
        pose.position = roi.center
        pose.orientation = roi.orientation
        
        # Transform the pose
        transformed_pose = do_transform_pose(pose, transform)
        
        # Update the roi with the transformed data
        roi.center = transformed_pose.position
        roi.orientation = transformed_pose.orientation
        roi.header.frame_id = target_frame
        return roi

    def is_overlapping(self, roi:Roi, obs:Observation):

        def get_bounding_box(roi):

            p1 = np.array([-roi.length/2, -roi.width/2, 0, 1])
            p2 = np.array([-roi.length/2, roi.width/2, 0, 1])
            p3 = np.array([roi.length/2, -roi.width/2, 0, 1])
            p4 = np.array([roi.length/2, roi.width/2, 0, 1])
            R = tf.transformations.quaternion_matrix([roi.orientation[0], roi.orientation[1], roi.orientation[2], roi.orientation[3]])
            T = np.eye(4)
            T[:3, 3] = [roi.center[0], roi.center[1], roi.center[2]]
            T[:3, :3] = R[:3, :3]
            p1 = np.dot(T, p1)[:2]
            p2 = np.dot(T, p2)[:2]
            p3 = np.dot(T, p3)[:2]
            p4 = np.dot(T, p4)[:2]
            x_coords = [p1[0], p2[0], p3[0], p4[0]]
            y_coords = [p1[1], p2[1], p3[1], p4[1]]
            return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

        x1_min, x1_max, y1_min, y1_max = get_bounding_box(roi)
        x2_min, x2_max, y2_min, y2_max = get_bounding_box(obs)

        return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

    def publish_markers(self, event):
        # Publish markers for visualization
        array = MarkerArray()
        for i, roi in enumerate(self.horiz_cavities):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "horiz_cavities"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = roi.center[0]
            marker.pose.position.y = roi.center[1]
            marker.pose.position.z = roi.depth / 2.0
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

        for i, roi in enumerate(self.vert_cavities):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "vert_cavities"
            marker.id = i + len(self.horiz_cavities)
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = roi.center[0]
            marker.pose.position.y = roi.center[1]
            marker.pose.position.z = roi.height / 2.0
            marker.scale.x = 0.05
            marker.scale.y = roi.width
            marker.scale.z = roi.height
            marker.color.a = 1.0
            marker.color.r = 0.5
            marker.color.g = 0.0
            marker.color.b = 0.5
            # Align the marker orientation to the plane model
            marker.pose.orientation.x = roi.orientation[0]
            marker.pose.orientation.y = roi.orientation[1]
            marker.pose.orientation.z = roi.orientation[2]
            marker.pose.orientation.w = roi.orientation[3]
            array.markers.append(marker)
        self.pub.publish(array)

    def run(self):
        rospy.Timer(rospy.Duration(1.0), self.publish_markers, oneshot=False)
        rospy.spin()


if __name__ == '__main__':
    try:
        cavity_map = CavityMap()
        cavity_map.run()
    except rospy.ROSInterruptException:
        pass
