#!/usr/bin/env python3

import numpy as np
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
import math
from cavity_detection_msgs.msg import Roi, RoiStamped, HorizontalObservation
from shapely.geometry import Polygon
from cavity_detection.helpers import *
import scipy

verbose = False

class HorizontalCluster:
    def __init__(self, id, initial_observation):
        lines = np.array(initial_observation.lines).reshape(-1, 4)
        if verbose: print(f"Initial lines: {lines}")
        self.anchor_point = np.array(lines[0][:2])
        self.id = id
        self.orientation = initial_observation.orientation
        self.length = initial_observation.length
        self.spacing = initial_observation.spacing
        self.height = initial_observation.height       
        self.cavities = []
        self.num_boards = len(lines)
        self.filled = False
        
        # --- EKF Initialization ---
        self.dim_x = 6
        self.ekf = ExtendedKalmanFilter(dim_x=self.dim_x, dim_z=4) # Set dim_z temporarily

        self.ekf.x = np.array([
            self.anchor_point[0], # px
            self.anchor_point[1], # py
            self.orientation, # theta
            self.length, # L
            self.spacing, # s
            self.height # height
        ])

        # Variances: Px, Py, Theta, Length, Spacing
        self.ekf.P = np.diag([0.5**2, 0.5**2, np.radians(15)**2, 0.3**2, 0.1**2, 0.1**2])

        # Static model
        self.ekf.F = np.identity(self.dim_x)
        # Process noise: How much might state drift between steps? Tune these variances!
        self.ekf.Q = np.diag([0.02**2, 0.02**2, np.radians(0.5)**2, 0.02**2, 0.01**2, 0.01**2])
        # self.ekf.Q = Q_discrete_white_noise(dim=self.dim_x, dt=1.,
        #                                     vars=[0.02**2, 0.02**2, np.radians(0.5)**2, 0.02**2, 0.01**2])

        # R1 for [theta_obs, L_obs, s_obs, height_obs]
        self.R1_intrinsics = np.diag([np.radians(5)**2, 0.1**2, 0.03**2, 0.1**2]) # Needs careful tuning!
        # R2 for position offset derived measurement [px_corrected, py_corrected]
        self.R2_position = np.diag([0.1**2, 0.1**2]) # Needs careful tuning!
    
    @property
    def estimated_lines(self):
        # Check if num_boards exists and is valid
        if not hasattr(self, 'num_boards') or self.num_boards <= 0:
            return np.empty((0, 4)) # Return empty array if no boards
        return generate_estimated_segments(self.ekf.x, self.num_boards)
    
    @property
    def width(self):
        """ Returns the width of the cavity based on the estimated state """
        # Assuming width is the spacing between boards
        return self.spacing * (self.num_boards - 1)

# Inside is_overlapping:
# cluster_lines = self.estimated_lines # Use the property
    def is_overlapping(self, observation):
        """ Checks if the bounding box of the cavity overlaps an observation """
        observed_lines = np.array(observation.lines).reshape(-1, 4)
        cluster_lines = self.estimated_lines
        observation_polygon = Polygon([tuple(observed_lines[0,:2]), tuple(observed_lines[-1,:2]), tuple(observed_lines[-1,2:]), tuple(observed_lines[0,2:])])
        cluster_polygon = Polygon([tuple(cluster_lines[0,:2]), tuple(cluster_lines[-1,:2]), tuple(cluster_lines[-1,2:]), tuple(cluster_lines[0,2:])])
        intersects = cluster_polygon.intersects(observation_polygon)
        length_difference = abs(self.length - observation.length)
        spacing_difference = abs(self.spacing - observation.spacing)
        angle_difference = self.orientation - observation.spacing
        orientation_difference = math.atan2(math.sin(angle_difference), math.cos(angle_difference)) 
        return intersects and length_difference < 1 and spacing_difference < 0.1 and orientation_difference < 1
    
    def boardwise_offset(self, observed_lines):
        """ Fuses two sets of line segments based on the estimated state vector """
        px, py, theta, L, s, _ = self.ekf.x
        observed_lines = np.array(observed_lines).reshape(-1, 4)
        origin = np.array([px, py])
        direction = np.array([math.cos(theta), math.sin(theta)])
        perp_direction = np.array([-math.sin(theta), math.cos(theta)]) 
        est_segments = generate_estimated_segments(self.ekf.x, self.num_boards)
        est_segments = est_segments.reshape(-1, 4)
        est_midpoints = (est_segments[:,:2] + est_segments[:,2:]) / 2.0
        observed_midpoints = (observed_lines[:,:2] + observed_lines[:,2:]) / 2.0
        mutual_pairs, offsets_xy, no_match = mutual_nearest_neighbors(est_midpoints, observed_midpoints)
        # Filter out pairs with large distances
        threshold = 0.3
        valid_pairs = np.linalg.norm(offsets_xy, axis=1) < threshold
        # Inside boardwise_offset:
        if np.any(valid_pairs):
            if verbose: print(f"Found {len(valid_pairs)} close matches out of {len(mutual_pairs)} total matches.")
            average_offset = np.mean(offsets_xy[valid_pairs], axis=0)
        else:
            # Handle case with no valid matches - maybe return zero offset or indicate failure?
            if verbose: print("No valid close matches found.")
            average_offset = np.zeros(2) # Example: assume no offset
            return average_offset

        # Apply the average offset to the estimated state vector
        new_origin = origin + average_offset
        before_start = origin - perp_direction * s 
        before_midpoint = before_start + direction * L / 2
        after_start = origin + perp_direction * s * (self.num_boards + 1)
        after_midpoint = after_start + direction * L / 2
        for i in no_match:
            if np.linalg.norm(observed_midpoints[i] - before_midpoint) < threshold:
                self.ekf.x[:2] = before_start
                self.num_boards += 1
                if verbose: print(f"Extending the ROI to include a new board at {new_origin}. Moving anchor.")
            elif np.linalg.norm(observed_midpoints[i] - after_midpoint) < threshold:
                self.num_boards += 1
                if verbose: print(f"Extending the ROI to include a new board at {after_start}")

        return average_offset

    def add_observation(self, observation):
        # --- 1. EKF Predict ---
        self.ekf.predict()
        x_pred = self.ekf.x_prior.copy()

        # --- 2. First EKF Update (Intrinsics) ---
        z1 = np.array([observation.orientation, observation.length, observation.spacing, observation.height])
        try:
            # Need to tell update the dimension of z1
            self.ekf.dim_z = 4
            self.ekf.update(z1, HJacobian=lambda x: H1_intrinsics_jacobian, # Use lambda for filterpy API
                           Hx=h1_intrinsics, R=self.R1_intrinsics, residual=residual_with_angle)
        except Exception as e:
             if verbose: print(f"EKF Update 1 failed for cluster {self.id}: {e}")
             # Failed update, maybe revert P? self.ekf.P = P_pred
             return False # Indicate fusion failed

        x_2 = self.ekf.x.copy() # State after intrinsic update

        # --- 3. Calculate Offset (and fuse lines in the process) ---
        # Generate segments based on the state *after* intrinsic update
        
        offset_xy = self.boardwise_offset(observation.lines)
        if verbose: print(f"Calculated offset: {offset_xy}")

        x_3 = self.ekf.x.copy()

        # --- 4. Second EKF Update (Position Correction via Offset) ---
        # Measurement z2 is the absolute position implied by the offset correction applied to the intermediate state's position
        z2 = np.array([x_3[0] + offset_xy[0], x_3[1] + offset_xy[1]])
        if verbose: print(f"Measurement 2 (z2): {z2}")
        try:
            # Need to tell update the dimension of z2
            self.ekf.dim_z = 2
            # We are updating the state x that already went through update 1
            self.ekf.update(z2, HJacobian=lambda x: H2_position_jacobian,
                           Hx=h2_position, R=self.R2_position) # Use standard residual
        except Exception as e:
             if verbose: print(f"EKF Update 2 failed for ROI {self.id}: {e}")             # Revert to intermediate state? self.ekf.x = x_intermediate; self.ekf.P = P_intermediate

             return False # Indicate fusion failed

        # --- 5. Store Final State in ROI ---
        self.update_roi_attributes_from_state() # Update helper attributes if any
        if verbose: print(f"Fusion complete for ROI {self.id}. Num boards: {self.num_boards}, Width: {self.width:.3f}")
        if verbose: print(f"---\n")

        return True # Indicate fusion succeeded

    def update_roi_attributes_from_state(self):
        """ Updates easily accessible attributes from the state vector x """
        self.anchor_point[0] = self.ekf.x[0]
        self.anchor_point[1] = self.ekf.x[1]
        self.orientation = self.ekf.x[2] 
        self.length = self.ekf.x[3]
        self.spacing = self.ekf.x[4]
        self.height = self.ekf.x[5]


class HorizontalCavity:
    def __init__(self, id, parent, front):
        self.id = id
        self.parent = parent
        self.front = front
        self.filled = False

class VerticalCluster:
    def __init__(self, id, initial_observation):
        self.id = id
        self.p1 = initial_observation.p1
        self.p2 = initial_observation.p2
        self.orientation = initial_observation.orientation
        self.num_cavities = 0
        self.cavities = []

class VerticalCavity:
    def __init__(self, id, parent, front):
        self.id = id
        self.parent = parent
        self.front = front
        self.filled = False

# --- Observation Class (Example) ---
class HorizontalClusterObservation:
    def __init__(self, theta, length, spacing):
        # Store observation data needed for fusion
        self.orientation = theta
        self.length = length
        self.spacing = spacing

        # Add pose if needed for initial ROI creation or direct measurement
        # self.pose = ...