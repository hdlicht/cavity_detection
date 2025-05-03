#!/usr/bin/env python3

import rospy
import tf
import tf2_ros
import tf2_geometry_msgs
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, PoseStamped, PointStamped
import numpy as np
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
import math
from matplotlib.path import Path
from scipy.spatial.transform import Rotation
from cavity_detection.ransac import ransac_line_fitting
from cavity_detection_msgs.msg import VerticalObservation, LogoObservation
from cavity_detection.rviz import vert_detector_markers
from tf.transformations import euler_from_quaternion
import copy

fx = 570.342
fy = 570.342
cx = 314.5
cy = 235.5

def line_equation(p1, p2):
    """Calculate the line equation (a, b, c) from two points."""
    if p2[0] - p1[0] == 0:
        a = 1
        b = 0
        c = p1[0] 
    else:
        a = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = -1
        c = p1[1] - a * p1[0]
    return (a, b, c)

def intersection(line1, line2):
    """Calculate the intersection point of two lines."""
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    d = a1 * b2 - a2 * b1
    if d == 0:
        return None
    x = (b1 * c2 - b2 * c1) / d
    y = (a2 * c1 - a1 * c2) / d
    return x, y

def transform_3d(points, trans, quat):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()  # Rotation matrix from quaternion
    T[:3, 3] = trans
    points = np.hstack((points, np.ones((points.shape[0], 1))))  # Convert to homogeneous coordinates
    new_points = T @ points.T  # Apply transformation
    new_points = new_points.T[:, :3]  # Convert back to Cartesian coordinates
    return new_points

def get_ray_directions(points, downsample=1):
    """Get the 3D points for an array of 2D pixels using the depth image."""
    pixels = np.arange(640)
    angles = -1*np.arctan2((pixels - cx), fx)  # Calculate angles in radians
    increment = -1*np.ones_like(pixels)
    for point in points:
        increment[np.logical_and(pixels>point[0], pixels<point[2])] = 1
    angles = angles[downsample::downsample]
    increment = increment[downsample::downsample]
    return angles, increment

def get_endpoint_rays(points):
    """Get the 3D points for an array of 2D pixels using the depth image."""
    pixels = np.arange(640)
    angle_pairs = []
    valid = []
    angles = -1*np.arctan2((pixels - cx), fx)  # Calculate angles in radians
    for point in points:
        a1 = angles[min(point[0], angles.shape[0]-1)] 
        a2 = angles[min(point[2], angles.shape[0]-1)]
        angle_pairs.append((a1, a2))
        v1 = point[0] > 50
        v2 = point[2] < 590
        valid.append((v1, v2))
    return angle_pairs, valid

class MapTracker:
    def __init__(self):
        rospy.init_node("map_tracker_node")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf.TransformListener()

        # --- Map and Grid Data ---
        self.map_metadata = None
        self.map_data = None # Stored obstacle map (NumPy)
        self.obstacle_threshold = 50 # Occupancy threshold for obstacles
        self.viewed_cells = set() # Cells that have been viewed

        self.vert_grid = OccupancyGrid() # Output grid message
        self.vert_grid_data = None # Output grid data (NumPy)
        self.vert_grid_initialized = False
        self.initial_vert_value = 0 # Initial value for vertical grid

        self.obs_grid = OccupancyGrid() # Output grid message
        self.obs_grid_data = None # Output grid data (NumPy)
        self.obs_grid_initialized = False
        self.initial_obs_value = 0 # Initial value for obstacle grid

        # --- Subscribers and Publishers ---
        self.occupancy_sub = rospy.Subscriber("/map", OccupancyGrid, self.occupancy_callback)
        self.intpair_sub = rospy.Subscriber("/vert_logo", LogoObservation, self.observation_callback)
        self.marker_pub = rospy.Publisher("/markers", MarkerArray, queue_size=1)
        self.vert_roi_pub = rospy.Publisher("/vert_roi", VerticalObservation, queue_size=2)
        self.vert_grid_pub = rospy.Publisher("/vertical_roi_map", OccupancyGrid, queue_size=1)
        self.obs_grid_pub = rospy.Publisher("/time_in_view_map", OccupancyGrid, queue_size=1)

        # rospy.Timer(rospy.Duration(0.5), self.explore_step)


    def world_to_map(self, wx, wy):
        """Converts world coordinates to map grid coordinates (float)."""
        if not self.map_metadata: return None, None
        mx = (wx - self.map_metadata.origin.position.x) / self.map_metadata.resolution
        my = (wy - self.map_metadata.origin.position.y) / self.map_metadata.resolution
        return mx, my

    def map_to_world(self, mx, my):
        """Converts map grid coordinates (center) to world coordinates."""
        if not self.map_metadata: return None, None
        wx = (mx + 0.5) * self.map_metadata.resolution + self.map_metadata.origin.position.x
        wy = (my + 0.5) * self.map_metadata.resolution + self.map_metadata.origin.position.y
        return wx, wy
    
    def publish_grids(self, event):
        """Publish the vertical and obstacle grids."""
        if self.vert_grid_initialized:
            self.vert_grid.data = self.vert_grid_data.flatten().tolist()
            self.vert_grid.header.stamp = rospy.Time.now()
            self.vert_grid_pub.publish(self.vert_grid)

        if self.obs_grid_initialized:
            self.obs_grid.data = self.obs_grid_data.flatten().tolist()
            self.obs_grid.header.stamp = rospy.Time.now()
            self.obs_grid_pub.publish(self.obs_grid)

    def occupancy_callback(self, grid_msg):
        """Process the occupancy grid and store map data."""
        # Store map data for later use (ray tracing)
        new_metadata = grid_msg.info
        new_data = np.array(grid_msg.data, dtype=np.int8).reshape(new_metadata.height, new_metadata.width)
        # Check if map metadata has changed significantly
        if self.map_metadata is None:
             self.map_metadata = new_metadata
             rospy.loginfo(f"Map metadata updated: {self.map_metadata.width}x{self.map_metadata.height} @ {self.map_metadata.resolution} m/cell")

        self.map_data = new_data # Update obstacle data

        # Initialize or update logo_grid to match the map
        if not self.vert_grid_initialized:
            rospy.loginfo("Initializing Vert Grid.")
            self.vert_grid.info = copy.deepcopy(self.map_metadata)
            # Initialize with neutral value
            self.vert_grid_data = np.full((self.map_metadata.height, self.map_metadata.width),
                                          self.initial_vert_value, dtype=np.int8)
            rospy.loginfo(f"Vert grid initialized with size: {self.vert_grid_data.shape}")
            self.vert_grid_initialized = True

        # Initialize or update logo_grid to match the map
        if not self.obs_grid_initialized:
            rospy.loginfo("Initializing Eyes-On Grid.")
            self.obs_grid.info = copy.deepcopy(self.map_metadata)
            # Initialize with neutral value
            self.obs_grid_data = np.full((self.map_metadata.height, self.map_metadata.width),
                                          self.initial_obs_value, dtype=np.int8)
            rospy.loginfo(f"Obs grid initialized with size: {self.obs_grid_data.shape}")
            self.obs_grid_initialized = True

    def observation_callback(self, msg):
        points = np.array(msg.points)
        points = points.reshape(-1, 4)
        try:
            pos, quat = self.tf_listener.lookupTransform("map", msg.header.frame_id, msg.header.stamp)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("Transform lookup failed.")
            return
        robot_heading = euler_from_quaternion(quat)[2]
        angles, increment = get_ray_directions(points)
        angle_pairs, valid = get_endpoint_rays(points)
        robot_x, robot_y = self.world_to_map(pos[0], pos[1])

        if robot_x is None or robot_y is None:
            rospy.logwarn("Invalid robot position for ray casting.")
            return
        
        for i, angle in enumerate(angles):
            hit = self.ray_cast(robot_x, robot_y, robot_heading, angle)
            if hit is not None:
                hit_my, hit_mx = hit
                self.vert_grid_data[hit_my, hit_mx] = np.clip(self.obs_grid_data[hit_my, hit_mx] + increment[i], 0, 100)
            else:
                rospy.logwarn("Logo detected but too far away from wall.")
        
        viewed_cells_list = list(self.viewed_cells)
        for cell in viewed_cells_list:
            cell_y, cell_x = cell
            if cell_x < self.obs_grid_data.shape[1] and cell_y < self.obs_grid_data.shape[0]:
                self.obs_grid_data[cell_y, cell_x] = np.clip(self.obs_grid_data[cell_y, cell_x] + 1, 0, 100)

        for i, pair in enumerate(angle_pairs):
            roi = VerticalObservation()
            roi.header = msg.header
            roi.header.frame_id = "map"
            roi.header.stamp = rospy.Time.now()
            map_cell_1 = self.ray_cast(robot_x, robot_y, robot_heading, pair[0])
            my1, mx1 = map_cell_1
            world_x1, world_y1 = self.map_to_world(mx1, my1)
            map_cell_2 = self.ray_cast(robot_x, robot_y, robot_heading, pair[1])
            my2, mx2 = map_cell_2
            world_x2, world_y2 = self.map_to_world(mx2, my2)
            orientation = np.arctan2(world_x2 - world_x1, world_y1 - world_y2)
            roi.orientation = orientation
            roi.p1 = [world_x1, world_y1] if valid[i][0] else [99., 99.]
            roi.p2 = [world_x2, world_y2] if valid[i][1] else [99., 99.]
            if valid[i][0] or valid[i][1]:
                self.vert_roi_pub.publish(roi)
                print(f"Published observation: {roi.p1} to {roi.p2} with orientation {roi.orientation}")


    def ray_cast(self, robot_x, robot_y, robot_heading, direction_relative, max_dist_meters=10.0):
            """
            Casts a ray from the robot's position using DDA and finds the first obstacle hit.

            Args:
                robot_x (float): Robot's X position in map grid coordinates.
                robot_y (float): Robot's Y position in map grid coordinates.
                robot_heading (float): Robot's orientation in map frame (radians, e.g., from yaw).
                direction_relative (float): Ray direction relative to robot_heading (radians).
                                            (e.g., from atan((px-cx)/fx) ).
                max_dist_meters (float): Maximum distance to cast the ray.

            Returns:
                tuple: (hit_mx, hit_my) integer map coordinates if an obstacle is hit.
                None: If no obstacle is hit within max distance or map bounds.
            """
            if self.map_data is None or self.map_metadata is None:
                # rospy.logwarn_throttle(5.0, "Ray casting attempted before map data is available.")
                return None

            # 1. Calculate Absolute Ray Direction
            absolute_ray_angle = robot_heading + direction_relative
            # Normalize angle (optional but good practice)
            # absolute_ray_angle = math.atan2(math.sin(absolute_ray_angle), math.cos(absolute_ray_angle))

            dir_x = math.cos(absolute_ray_angle)
            dir_y = math.sin(absolute_ray_angle)

            # --- DDA Setup ---
            # Check for zero direction vector
            if abs(dir_x) < 1e-9 and abs(dir_y) < 1e-9:
                # rospy.logwarn_throttle(5.0, "Ray casting with near-zero direction vector.")
                return None

            resolution = self.map_metadata.resolution
            max_dist_cells = int(max_dist_meters / resolution)

            # Start cell (integer map coords)
            start_map_x_int = int(math.floor(robot_x))
            start_map_y_int = int(math.floor(robot_y))

            # Current integer cell coords
            current_map_x = start_map_x_int
            current_map_y = start_map_y_int

            # Distance step required to cross one cell boundary (in units of ray length)
            epsilon = 1e-9
            t_delta_x = float('inf') if abs(dir_x) < epsilon else abs(1.0 / dir_x)
            t_delta_y = float('inf') if abs(dir_y) < epsilon else abs(1.0 / dir_y)

            # Step direction (1 or -1)
            step_x = 1 if dir_x >= 0 else -1
            step_y = 1 if dir_y >= 0 else -1

            # Distance to nearest cell boundary intersection (in units of ray length)
            if dir_x >= 0:
                t_max_x = (math.floor(robot_x) + 1.0 - robot_x) * t_delta_x
            else:
                t_max_x = (robot_x - math.floor(robot_x)) * t_delta_x

            if dir_y >= 0:
                t_max_y = (math.floor(robot_y) + 1.0 - robot_y) * t_delta_y
            else:
                t_max_y = (robot_y - math.floor(robot_y)) * t_delta_y

            # Make sure initial t_max are non-negative
            if abs(t_max_x) < epsilon: t_max_x = t_delta_x
            if abs(t_max_y) < epsilon: t_max_y = t_delta_y

            dist_traveled_cells = 0
            viewed_cells = set()

            # --- DDA Stepping Loop ---
            while dist_traveled_cells < max_dist_cells:
                # Decide whether to step in X or Y
                if t_max_x < t_max_y:
                    current_map_x += step_x # Step X
                    dist_traveled_cells = t_max_x # This is the distance *to* the boundary just crossed
                    t_max_x += t_delta_x
                else:
                    current_map_y += step_y # Step Y
                    dist_traveled_cells = t_max_y
                    t_max_y += t_delta_y

                viewed_cells.add((current_map_y, current_map_x))

                # Check map bounds
                if not (0 <= current_map_y < self.map_metadata.height and 0 <= current_map_x < self.map_metadata.width):
                    self.viewed_cells.update(viewed_cells)
                    return None # Ray left map

                # Check obstacle cost
                try:
                    cost = self.map_data[current_map_y, current_map_x] # NumPy [row, col] = [y, x]
                    if cost >= self.obstacle_threshold:
                        self.viewed_cells.update(viewed_cells)
                        return (current_map_y, current_map_x) # Hit an obstacle
                except IndexError:
                    rospy.logwarn_throttle(5.0, f"IndexError during ray casting cost check at ({current_map_x},{current_map_y}).")
                    self.viewed_cells.update(viewed_cells)
                    return None
                
                dist_traveled_cells += 1

            self.viewed_cells.update(viewed_cells)
            return None # Reached max distance without hit

            
if __name__ == "__main__":
    node = MapTracker()
    rospy.Timer(rospy.Duration(1.0), node.publish_grids)
    rospy.spin()

