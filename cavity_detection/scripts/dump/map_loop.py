#!/usr/bin/env python3

import rospy
import tf2_ros
import tf2_geometry_msgs
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, PoseStamped, PointStamped
import numpy as np
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import math
from matplotlib.path import Path
from cavity_detection.ransac import ransac_line_fitting
from cavity_detection_msgs.msg import VerticalObservation
from cavity_detection.helpers import transform_2d, transform_3d, invert_2d_transform, points_from_grid
from cavity_detection.rviz import vert_detector_markers


K_rgb = np.array([[570.342, 0.0,     314.5],
                [0.0,     570.342, 235.5],
                [0.0,     0.0,     1.0]])

class TriangleOccupancyHighlighter:
    def __init__(self):
        rospy.init_node("map_looper")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.occupancy_sub = rospy.Subscriber("/map", OccupancyGrid, self.occupancy_callback)
        self.intpair_sub = rospy.Subscriber("/vert_roi", VerticalObservation, self.observation_callback)
        self.marker_pub = rospy.Publisher("/markers", MarkerArray, queue_size=1)

        self.occupied_world_points = None
        self.laser_points_map = None
        self.transformed_vertices = None

    def transform_triangle_to_map(self, triangle_vertices_camera, frame, stamp=rospy.Time(0)):
        """Transform triangle vertices from camera optical frame to map frame."""
        try:
            tf = self.tf_buffer.lookup_transform("map", frame, rospy.Time(0), rospy.Duration(1.0))
            trans = np.array([tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z])
            quat = np.array([tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w])
            new_points = transform_3d(triangle_vertices_camera, trans, quat)
            return new_points

        except Exception as e:
            rospy.logwarn("Transform lookup failed: %s", str(e))
            return None

    def occupancy_callback(self, grid_msg):
        """Process the occupancy grid to find cells inside the transformed triangle."""

        # Extract grid metadata
        width = grid_msg.info.width
        height = grid_msg.info.height
        resolution = grid_msg.info.resolution
        origin_x = grid_msg.info.origin.position.x
        origin_y = grid_msg.info.origin.position.y
        data = np.array(grid_msg.data).reshape(height, width)

        occupied_cells = np.argwhere(data >= 50)  # Cells with probability >= 50

        # Convert grid indices to world coordinates
        self.occupied_world_points = np.array([
            [x * resolution + origin_x, y * resolution + origin_y] 
            for y, x in occupied_cells
        ])

    def observation_callback(self, msg):
        numbers = np.array(msg.points)
        numbers = numbers.reshape(-1, 4)
        marker_array = MarkerArray()

        for i in range(len(numbers)):
            left_pixel = numbers[i][0]
            right_pixel = numbers[i][2]
            depth = 5.0
            left = (left_pixel - K_rgb[0, 2]) / K_rgb[0, 0] * depth
            right = (right_pixel - K_rgb[0, 2]) / K_rgb[0, 0] * depth
            # Define triangle in the camera optical frame (Z-forward, X-right, Y-down)
            triangle_vertices_camera = np.array([
                [0, 0, 0],      # Camera origin
                [left, 0, depth],  # Left boundary at 10m
                [right, 0, depth]    # Right boundary at 10m
            ])

            triangle_vertices_map = self.transform_triangle_to_map(triangle_vertices_camera, msg.header.frame_id, msg.header.stamp)

            if self.occupied_world_points is not None and triangle_vertices_map is not None:
                # Filter points inside the triangle
                triangle_polygon = Path(triangle_vertices_map[:,:2])
                inside_points = self.occupied_world_points[triangle_polygon.contains_points(self.occupied_world_points)]
                if len(inside_points) < 3:
                    rospy.logwarn("Not enough points inside the triangle.")
                    continue
                line_model, inliers = ransac_line_fitting(inside_points, 0.02, 100)
                if line_model is None:
                    rospy.logwarn("No line model found.")
                    continue
                # Calculate the line equation
                a, b, c = line_model
                # calculate the line equation of the left edge of the triangle
                a_left = (triangle_vertices_map[1][1] - triangle_vertices_map[0][1]) / (triangle_vertices_map[1][0] - triangle_vertices_map[0][0])
                b_left = -1
                c_left = triangle_vertices_map[0][1] - a_left * triangle_vertices_map[0][0]
                d_left = a * b_left - b * a_left
                # find where left intersects the line
                x1 = -(c*b_left - c_left*b) / d_left
                y1 = -(a*c_left - a_left*c) / d_left
                # calculate the line equation of the right edge of the triangle
                a_right = (triangle_vertices_map[2][1] - triangle_vertices_map[0][1]) / (triangle_vertices_map[2][0] - triangle_vertices_map[0][0])
                b_right = -1
                c_right = triangle_vertices_map[0][1] - a_right * triangle_vertices_map[0][0]
                d_right = a * b_right - b * a_right
                # find where right intersects the line
                x2 = -(c*b_right - c_right*b) / d_right
                y2 = -(a*c_right - a_right*c) / d_right
                
                # Publish highlighted cells
                triangle_marker, points_marker, p1_marker, p2_marker = vert_detector_markers(i, inside_points, triangle_vertices_map, x1, y1, x2, y2)

                # Add markers to the marker array
                marker_array.markers.append(p1_marker)
                marker_array.markers.append(p2_marker)
                marker_array.markers.append(triangle_marker)
                marker_array.markers.append(points_marker)

        if len(marker_array.markers) == 0:
            rospy.logwarn("No markers to publish.")
            return
        
        self.marker_pub.publish(marker_array)

if __name__ == "__main__":
    node = TriangleOccupancyHighlighter()
    rospy.spin()

