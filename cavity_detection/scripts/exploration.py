#!/usr/bin/env python3
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from scipy.ndimage import convolve, binary_dilation
from sklearn.cluster import DBSCAN
import numpy as np
from cavity_detection.map_wrapper import MapWrapper
import tf
from tf import TransformListener
from tf.transformations import quaternion_from_euler
from enum import Enum
from cavity_detection_msgs.msg import HorizontalObservation, VerticalObservation
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.srv import GetPlan, GetPlanRequest, GetPlanResponse

class RobotState(Enum):
    EXPLORING = 0
    INVESTIGATING = 1
    MOVING = 2
    DONE = 3
(0.14361082372528244, 0.7081965360675699)
class FrontierExplorer:
    def __init__(self):
        rospy.init_node('naive_frontier_explorer')
        rospy.wait_for_service('/move_base/make_plan')
        self.make_plan_srv = rospy.ServiceProxy('/move_base/make_plan', GetPlan)
        self.robot_state = RobotState.EXPLORING
        self.tf_listener = TransformListener()
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        self.horizontal_sub = rospy.Subscriber("/horiz_roi", HorizontalObservation, self.horiz_callback)
        self.vertical_sub = rospy.Subscriber("/vert_roi", VerticalObservation, self.vert_callback)
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        self.marker_pub = rospy.Publisher("/frontier_markers", MarkerArray, queue_size=1)
        rospy.Timer(rospy.Duration(0.5), self.explore_step)
        self.map = None
        self.frontiers = None
        self.frontier_sizes = None
        self.block_markers = False

    def map_callback(self, msg):
        print("Received map.")
        self.map = MapWrapper(msg)
        self.frontiers, self.frontier_sizes = self.map.get_frontiers()

    def horiz_callback(self, msg):
        # Placeholder for horizontal observation logic
        rospy.loginfo("Received horizontal observation.")

    def vert_callback(self, msg):
        # Placeholder for vertical observation logic
        rospy.loginfo("Received vertical observation.")

    def get_current_pose(self):
        try:
            now = rospy.Time(0)
            self.tf_listener.waitForTransform("map", "base_link", now, rospy.Duration(1.0))
            pos, quat = self.tf_listener.lookupTransform("map", "base_link", now)
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = pos[0]
            pose.pose.position.y = pos[1]
            pose.pose.position.z = pos[2]
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            return pose
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Could not get current pose.")
            return None
        
    def make_goal(self, start, wx, wy):
        sx = start.pose.position.x
        sy = start.pose.position.y
        theta = np.arctan2(wx - sx, wy - sy)
        print("Theta:", np.degrees(theta))
        quat = quaternion_from_euler(0, 0, theta)
        print("Quaternion:", quat)
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = wx
        goal.pose.position.y = wy
        goal.pose.position.z = 0.0
        goal.pose.orientation.x = quat[0]
        goal.pose.orientation.y = quat[1]
        goal.pose.orientation.z = quat[2]
        goal.pose.orientation.w = quat[3]
        return goal
        
    def move_base_plan(self, wx, wy, tolerance=0.5):
        req = GetPlanRequest()
        start = self.get_current_pose()

        req.start = start
        req.goal = self.make_goal(start, wx, wy)
        req.tolerance = tolerance
        try:
            plan = self.make_plan_srv(req)
            if plan is None or len(plan.plan.poses) == 0:
                rospy.loginfo("No valid path found to goal.")
                return None
            return plan.plan.poses  
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to call make_plan: {e}")
            return None

    def rank_frontiers(self):
        if self.frontiers is None or len(self.frontiers) == 0:
            rospy.loginfo("No frontiers available.")
            return
        try:
            now = rospy.Time(0)
            self.tf_listener.waitForTransform("map", "base_link", now, rospy.Duration(1.0))
            pos, quat = self.tf_listener.lookupTransform("map", "base_link", now)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Transform lookup failed.")
            return
        print("Current position:", pos)
        print(self.frontiers)
        distances = [np.linalg.norm(frontier - np.array(pos[:2])) for frontier in self.frontiers]
        print(distances)
        ranks = np.argsort(distances)
        print(ranks)
        ranked_frontiers = [t for _, t in reversed(sorted(zip(ranks, self.frontiers)))]
        print(ranked_frontiers)
        return ranked_frontiers

    def explore_step(self, event):
        if self.robot_state == RobotState.EXPLORING:
            if self.map is None:
                rospy.logwarn("Map not available yet.")
                return
            rospy.loginfo("Exploring frontiers.")
            ranked_frontiers = self.rank_frontiers()
            if ranked_frontiers is not None:
                marker_array = MarkerArray()
                for i, frontier in enumerate(ranked_frontiers):
                    wx, wy = frontier
                    rospy.loginfo(f"Exploring frontier {i}: ({wx}, {wy})")
                    # Check if the goal is reachable
                    path = self.move_base_plan(wx, wy)
                    reachable = path is not None and len(path) > 0
                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.header.stamp = rospy.Time.now()
                    marker.ns = "frontiers"
                    marker.id = i
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD
                    marker.pose.position.x = frontier[0]
                    marker.pose.position.y = frontier[1]
                    marker.pose.position.z = 0.0
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 0.5
                    marker.scale.y = 0.5
                    marker.scale.z = 0.5
                    marker.color.r = not reachable
                    marker.color.g = reachable
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                    marker_array.markers.append(marker)
                    if reachable:
                        rospy.loginfo(f"Reachable frontier at ({wx}, {wy}). {len(path)} steps.")
                        for i, pose in enumerate(path):
                            if i % 10 == 0:
                                q = pose.pose.orientation
                                print(f"Marker {i}: quat=({q.x:.2f}, {q.y:.2f}, {q.z:.2f}, {q.w:.2f})")
                                marker = Marker()
                                marker.header.frame_id = "map"
                                marker.header.stamp = rospy.Time.now()
                                marker.ns = "path"
                                marker.id = i
                                marker.type = Marker.ARROW
                                marker.action = Marker.ADD
                                marker.pose.position.x = pose.pose.position.x
                                marker.pose.position.y = pose.pose.position.y
                                marker.pose.position.z = 0.5
                                marker.pose.orientation.x = pose.pose.orientation.x
                                marker.pose.orientation.y = pose.pose.orientation.y  
                                marker.pose.orientation.z = pose.pose.orientation.z
                                marker.pose.orientation.w = pose.pose.orientation.w
                                marker.scale.x = 0.3
                                marker.scale.y = 0.1
                                marker.scale.z = 0.1
                                marker.color.r = 0.0
                                marker.color.g = 0.0
                                marker.color.b = 1.0
                                marker.color.a = 1.0
                                marker_array.markers.append(marker)
                        self.marker_pub.publish(marker_array)
                        # input("Press Enter to go to the goal...")
                        # start = self.get_current_pose()
                        # goal = self.make_goal(start, wx, wy)
                        # self.goal_pub.publish(goal)
                        break
                # make a path to the goal that has the robot look at the nearest unobserved wall
                self.robot_state = RobotState.MOVING
            else:
                rospy.loginfo("No valid frontiers to explore.")
                self.robot_state = RobotState.DONE

        elif self.robot_state == RobotState.MOVING:
            rospy.loginfo("Moving to goal.")
            # While moving to goal, listen for observations
            # if observation comes in, switch to investigating the area of the observation
            # When robot reaches goal, switch back to exploring
            self.robot_state = RobotState.INVESTIGATING

        elif self.robot_state == RobotState.INVESTIGATING:
            # Placeholder for investigation logic
            rospy.loginfo("Investigating current location.")
            self.robot_state = RobotState.EXPLORING

if __name__ == "__main__":
    explorer = FrontierExplorer()
    rospy.spin()
