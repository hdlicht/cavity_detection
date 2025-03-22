#!/usr/bin/env python3

import rospy
import tf2_ros
import tf2_geometry_msgs
from visualization_msgs.msg import Marker
from std_msgs.msg import Int16MultiArray
from geometry_msgs.msg import Point, PoseStamped
import numpy as np
from nav_msgs.msg import OccupancyGrid

K_rgb = np.array([[570.342, 0.0,     314.5], 
                [0.0,     570.342, 235.5],
                [0.0,     0.0,     1.0]])


class TriangleOccupancyHighlighter:
    def __init__(self):
        rospy.init_node("triangle_occupancy_highlighter")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.occupancy_sub = rospy.Subscriber("/map", OccupancyGrid, self.occupancy_callback)
        self.intpair_sub = rospy.Subscriber("/int_pair", Int16MultiArray, self.intpair_callback)
        self.marker_pub = rospy.Publisher("/highlighted_cells", Marker, queue_size=1)
        self.triangle_pub = rospy.Publisher("/triangle", Marker, queue_size=1)

        self.occupied_world_points = None

        self.transformed_vertices = None

    def transform_triangle_to_map(self, triangle_vertices_camera):
        """Transform triangle vertices from camera optical frame to map frame."""
        try:
            trans = self.tf_buffer.lookup_transform("map", "camera_rgb_optical_frame", rospy.Time(0), rospy.Duration(1.0))

            transformed_points = []
            for pt in triangle_vertices_camera:
                pose = PoseStamped()
                pose.header.frame_id = "camera_rgb_optical_frame"
                pose.header.stamp = rospy.Time(0)
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = pt
                transformed_pose = tf2_geometry_msgs.do_transform_pose(pose, trans)
                transformed_points.append([
                    transformed_pose.pose.position.x,
                    transformed_pose.pose.position.y
                ])

            return np.array(transformed_points)

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

    def intpair_callback(self, msg):

        left_pixel, right_pixel = msg.data
        depth = 8.0
        left = (left_pixel - K_rgb[0, 2]) / K_rgb[0, 0] * depth
        right = (right_pixel - K_rgb[0, 2]) / K_rgb[0, 0] * depth

        # Define triangle in the camera optical frame (Z-forward, X-right, Y-down)
        triangle_vertices_camera = np.array([
            [0, 0, 0],      # Camera origin
            [left, 0, depth],  # Left boundary at 10m
            [right, 0, depth]    # Right boundary at 10m
        ])

        triangle_vertices_map = self.transform_triangle_to_map(triangle_vertices_camera)

        if self.occupied_world_points is not None and triangle_vertices_map is not None:
            # Filter points inside the triangle
            inside_points = [pt for pt in self.occupied_world_points if self.is_point_inside_triangle(pt, triangle_vertices_map)]

            # Publish highlighted cells
            self.publish_marker(inside_points, triangle_vertices_map)


    def is_point_inside_triangle(self, pt, triangle_vertices):
        """Check if a 2D point is inside the triangle using cross-product method."""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        v1, v2, v3 = triangle_vertices
        d1 = sign(pt, v1, v2)
        d2 = sign(pt, v2, v3)
        d3 = sign(pt, v3, v1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)  # True if all signs are the same

    def publish_marker(self, points, vertices):
        """Publishes a marker containing both the triangle and highlighted cells."""

        # Triangle edges (LINE_STRIP)
        triangle_marker = Marker()
        triangle_marker.header.frame_id = "map"
        triangle_marker.header.stamp = rospy.Time.now()
        triangle_marker.ns = "triangle"
        triangle_marker.id = 1
        triangle_marker.type = Marker.LINE_STRIP
        triangle_marker.action = Marker.ADD
        triangle_marker.scale.x = 0.05  # Line thickness
        triangle_marker.color.r = 0.0
        triangle_marker.color.g = 1.0
        triangle_marker.color.b = 0.0
        triangle_marker.color.a = 1.0

        for v in vertices:
            p = Point()
            p.x, p.y = v
            p.z = 0
            triangle_marker.points.append(p)

        # Close the triangle
        triangle_marker.points.append(triangle_marker.points[0])

        # Highlighted occupied cells (POINTS)
        points_marker = Marker()
        points_marker.header.frame_id = "map"
        points_marker.header.stamp = rospy.Time.now()
        points_marker.ns = "highlighted_cells"
        points_marker.id = 2
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

        # Publish both markers
        self.triangle_pub.publish(triangle_marker)
        self.marker_pub.publish(points_marker)

if __name__ == "__main__":
    node = TriangleOccupancyHighlighter()
    rospy.spin()

