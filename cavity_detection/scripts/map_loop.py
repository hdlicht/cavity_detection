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
from cavity_detection_msgs.msg import LogoObservation
from cavity_detection.helpers import transform_2d, transform_3d, invert_2d_transform, points_from_grid


K_rgb = np.array([[570.342, 0.0,     314.5],
                [0.0,     570.342, 235.5],
                [0.0,     0.0,     1.0]])

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

class TriangleOccupancyHighlighter:
    def __init__(self):
        rospy.init_node("map_looper")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.occupancy_sub = rospy.Subscriber("/map", OccupancyGrid, self.occupancy_callback)
        self.intpair_sub = rospy.Subscriber("/vert_logo", LogoObservation, self.observation_callback)
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
            depth = 3.0
            left = (left_pixel - K_rgb[0, 2]) / K_rgb[0, 0] * depth
            right = (right_pixel - K_rgb[0, 2]) / K_rgb[0, 0] * depth
            # Define triangle in the camera optical frame (Z-forward, X-right, Y-down)
            triangle_vertices_camera = np.array([
                [0, 0, 0],      # Camera origin
                [depth, left, 0],  # Left boundary at 10m
                [depth, right, 0]    # Right boundary at 10m
            ])

            triangle_vertices_map = self.transform_triangle_to_map(triangle_vertices_camera, msg.header.frame_id, msg.header.stamp)

            if self.occupied_world_points is not None and triangle_vertices_map is not None:
                # Filter points inside the triangle
                triangle_polygon = Path(triangle_vertices_map[:,:2])
                print(triangle_polygon.vertices)
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
        print("publishing")
        self.marker_pub.publish(marker_array)

if __name__ == "__main__":
    node = TriangleOccupancyHighlighter()
    rospy.spin()

