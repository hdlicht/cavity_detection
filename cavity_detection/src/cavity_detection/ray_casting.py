import math
import numpy as np
import rospy

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