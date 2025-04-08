#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import threading
import numpy as np
from sklearn.cluster import DBSCAN
import threading
from cavity_detection.ransac import ransac_plane_fitting
from cavity_detection_msgs.msg import RoiStamped, HorizontalObservation
from geometry_msgs.msg import Point, Quaternion
from visualization_msgs.msg import Marker
import time
from random import randint
from math import atan2

# # Global shared buffer
# data_buffer = {"rgb": None, "depth": None}
# buffer_lock = threading.Lock()

# Desired frequency for fusion (Hz)
RUN_RATE = 5  
SIZE_FACTOR = 1.0
GROUND_NORMAL = np.array([0, 0, 1])  
MIN_LINES_FOR_PARALLEL_CLUSTER = 3
MIN_LINES_FOR_COLINEAR_CLUSTER = 3
SPACING_TOLERANCE_RATIO = 0.15 # e.g., +/- 15%
MIN_SIGNIFICANT_DISTANCE = 0.01 # e.g., 1 cm minimum gap to consider for spacing
PARALLEL_EPSILON = 0.1 # Epsilon for DBSCAN clustering of parallel lines
COLINEAR_EPSILON = 10 # Epsilon for DBSCAN clustering of collinear lines

K_rgb = np.array([[570.342, 0.0,     314.5], 
                [0.0,     570.342, 235.5],
                [0.0,     0.0,     1.0]])

K_depth = np.array([[570.342, 0.0,     319.5],
                    [0.0,     570.342, 239.5],
                    [0.0,     0.0,     1.0]])

# K_rgb = np.array([[910.1351318359375, 0.0, 643.9998779296875], 
#                   [0.0, 910.0550537109375, 370.72381591796875], 
#                   [0.0, 0.0, 1.0]])

# K_depth = np.array([[910.1351318359375, 0.0, 643.9998779296875], 
#                     [0.0, 910.0550537109375, 370.72381591796875], 
#                     [0.0, 0.0, 1.0]])

K_scaler = np.array([[SIZE_FACTOR], [SIZE_FACTOR], [1]])
K_depth *= K_scaler
K_rgb *= K_scaler

# Transformation matrix from depth to RGB
T_depth_rgb = np.array([
    [1, 0, 0, -0.025], 
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1] ])

T_camera_world = np.array([
    [0,  0,  1,  0],  
    [-1,  0, 0,  0],  
    [0,  -1,  0,  0.7], 
    [0,  0,  0,  1] ])

models = []
average_model = None
calibration_count = 0
runtimes = []

def transform_2d(point, origin, angle):
    T = np.array([[np.cos(angle), -np.sin(angle), origin[0]], 
                  [np.sin(angle), np.cos(angle), origin[1]], 
                  [0, 0, 1]])
    point = np.array([point[0], point[1], 1])
    new_point = T @ point
    return (new_point[0], new_point[1])

def normalize_lines(lines):
    # Ensure lines have consistent ordering: (x1, y1, x2, y2) where x1 < x2 or (x1 == x2 and y1 < y2)
    x1, y1, x2, y2 = lines[:, 0], lines[:, 1], lines[:, 2], lines[:, 3]
    swap = (x1 > x2) | ((x1 == x2) & (y1 > y2))
    normalized = lines.copy()
    normalized[swap] = lines[swap][:, [2, 3, 0, 1]]
    return normalized

def get_depth(depth_image, x, y):
    """Get the depth value at pixel (x, y) from the depth image."""
    return depth_image[y, x]

def get_3d_points(depth_image, points_2d):
    """Get the 3D points for an array of 2D pixels using the depth image."""
    zs = depth_image[points_2d[:, 1], points_2d[:, 0]]
    valid = np.logical_and(zs > 0, zs < 5) # Filter out invalid depth values
    xs = (points_2d[:, 0] - K_rgb[0, 2]) / K_rgb[0, 0] * zs
    ys = (points_2d[:, 1] - K_rgb[1, 2]) / K_rgb[1, 1] * zs
    points_3d = np.vstack((xs, ys, zs)).T
    return points_3d[valid]

def detect(depth_image, time_stamp):
    ct = 0 # What is ct used for? Seems unused.
    """Periodically process the fusion of RGB and Depth images."""
    # global pub, models, calibration_count, average_model, runtimes - Remove unused globals if possible
    global pub2, pub3, bridge, T_camera_world, SIZE_FACTOR # Keep necessary globals/constants
    start_time = time.time()

    depth_image = cv2.resize(depth_image, (0, 0), fx=SIZE_FACTOR, fy=SIZE_FACTOR)
    depth_image = depth_image # / 1000.
    H, W = depth_image.shape[:2]
    # Consider making ROI configurable or dynamically determined
    bottom = [0, H//2, W, H]
    # bottom = [0, 0, W, H//2] # Alternative ROI

    x, y = np.meshgrid(range(bottom[0], bottom[2]), range(bottom[1], bottom[3]))
    points_2d = np.vstack([x.ravel(), y.ravel()]).T
    
    # Handle potential errors in 3D conversion
    try:
        points_3d_camera = get_3d_points(depth_image, points_2d)
    except Exception as e:
        #print(f"Error in get_3d_points: {e}")
        return

    # Filter out invalid points (e.g., NaN, Inf) right after conversion
    valid_3d_indices = ~np.isnan(points_3d_camera).any(axis=1) & ~np.isinf(points_3d_camera).any(axis=1)
    if not np.any(valid_3d_indices):
        #print("No valid 3D points found in ROI.")
        return
    points_3d_camera = points_3d_camera[valid_3d_indices]

    # Transform the points to the world frame
    points_world = np.hstack((points_3d_camera, np.ones((points_3d_camera.shape[0], 1))))
    points_world = np.dot(T_camera_world, points_world.T).T
    points_world = points_world[:, :3] # Keep only x, y, z
    max_z = np.nanmax(points_world[:, 2])
    min_z = np.nanmin(points_world[:, 2])
    #print(f"Max Z: {max_z:.3f}, Min Z: {min_z:.3f}")

    # --- Plane Finding Section ---
    # Using hardcoded ground plane for now
    average_model = [0, 0, 1, 0] # Represents z = 0.05 plane (ax+by+cz+d=0 -> cz = -d -> z = -d/c)
    a, b, c, d_ground = average_model # d_ground is negative offset

    # Calculate distances of all points to potential horizontal planes
    best_inliers = []
    best_count = 0
    best_d = 0 # This 'd' is the constant in z = d (or ax+by+cz-d=0 where a=b=0, c=1)

    # Search for a plane parallel to ground (z=const) with most inliers
    # Adjust search range [0.1, 0.6] based on expected object heights
    for height_offset in np.linspace(0.15, 0.5, 20): # height_offset = z value
        # deviation = np.abs(a * points_world[:, 0] + b * points_world[:, 1] + c * points_world[:, 2] + d_plane)
        # For plane z = height_offset (or 0x+0y+1z-height_offset = 0), deviation is simpler:
        deviation = np.abs(points_world[:, 2] - height_offset)
        inlier_indices = np.where(deviation < 0.0127)[0] # Threshold for planarity

        if len(inlier_indices) > best_count:
            best_count = len(inlier_indices)
            best_inliers_indices = inlier_indices
            best_d = height_offset # Store the height (z-value) of the best plane

    if best_count < 10: # Need a minimum number of points to define the surface
        #print("Not enough inliers found on any horizontal plane.")
        return

    inlier_points_world = points_world[best_inliers_indices]
    #print(f"Found {best_count} inliers on plane z = {best_d:.3f}m")
    if inlier_points_world.shape[0] == 0:
        #print("Inlier points array is empty unexpectedly.")
        return
    min_xyz_inliers = np.nanmin(inlier_points_world, axis=0)

    # Shift inlier points relative to their own minimum
    xy_relative = inlier_points_world[:, :2] - min_xyz_inliers[:2] # Use only X, Y for 2D image

    # Handle cases where dimensions might be zero or negative after subtraction/scaling
    max_x_rel = np.max(xy_relative[:, 0]) if xy_relative.shape[0] > 0 else 0
    max_y_rel = np.max(xy_relative[:, 1]) if xy_relative.shape[0] > 0 else 0

    W_img = int(max_x_rel * 100) + 1 # Add 1 for safety margin
    H_img = int(max_y_rel * 100) + 1

    if W_img <= 0 or H_img <= 0:
         print(f"Invalid image dimensions calculated: W={W_img}, H={H_img}. Skipping.")
         return

    # Create top-down 2D image
    tops_2d = np.zeros((H_img, W_img), dtype=np.uint8)
    for x_rel, y_rel in xy_relative:
        # Map to image coords, ensuring they are within bounds
        col = min(max(0, int(x_rel * 100)), W_img - 1)
        row = min(max(0, int(y_rel * 100)), H_img - 1)
        tops_2d[row, col] = 255

    # Apply morphological operations
    tops_2d = cv2.dilate(tops_2d, None, iterations=2)
    tops_2d = cv2.erode(tops_2d, None, iterations=1)

    try:
        # Convert to color for visualization
        tops_2d_col = cv2.cvtColor(tops_2d, cv2.COLOR_GRAY2BGR)
        ros_image = bridge.cv2_to_imgmsg(tops_2d_col, encoding="bgr8")
        pub3.publish(ros_image)
    except Exception as e:
        print(f"Error publishing tops_2d image: {e}")
    
    # --- Line Detection ---
    # Adjust Hough parameters as needed
    all_lines = cv2.HoughLinesP(tops_2d, 1, np.pi/180, threshold=100, minLineLength=70, maxLineGap=20)
    if all_lines is None:
        #print("No lines detected by Hough Transform.")
        return
    all_lines = all_lines.reshape(-1, 4) # Ensure shape is (N, 4)

    # Normalize lines (ensure this function handles potential zero-length lines safely)
    try:
        all_lines_normalized = normalize_lines(all_lines)
        if np.isnan(all_lines_normalized).any():
            #print("Warning: NaN values found after normalizing lines. Trying to proceed.")
            all_lines_normalized = all_lines_normalized[~np.isnan(all_lines_normalized).any(axis=1)]
    except Exception as e:
        #print(f"Error during line normalization: {e}")
        return

    # Calculate line angles, handling potential NaNs from normalization
    dx = all_lines_normalized[:, 2] - all_lines_normalized[:, 0]
    dy = all_lines_normalized[:, 3] - all_lines_normalized[:, 1]
    line_angles = np.arctan2(dy, dx)

    # --- Parallel Line Clustering ---
    if len(line_angles) < MIN_LINES_FOR_PARALLEL_CLUSTER:
        #print(f"Not enough lines ({len(line_angles)}) found for parallel clustering.")
        return

    angle_clustering = DBSCAN(eps=0.1, min_samples=MIN_LINES_FOR_PARALLEL_CLUSTER).fit(line_angles.reshape(-1, 1))
    labels_angle = angle_clustering.labels_

    unique_labels, counts = np.unique(labels_angle[labels_angle != -1], return_counts=True) # Ignore noise label -1

    if len(counts) == 0:
        # #print("No non-noise parallel line clusters found.")
        return # No suitable cluster found

    # Find the label corresponding to the cluster with the most lines
    best_label_index = np.argmax(counts)
    best_angle_label = unique_labels[best_label_index]
    num_lines_in_best_cluster = counts[best_label_index]

    #print(f"Found {len(unique_labels)} parallel clusters. Processing largest (Label {best_angle_label}) with {num_lines_in_best_cluster} lines.")

    # Get indices for the lines belonging to the best cluster
    indices_best_parallel = np.where(labels_angle == best_angle_label)[0]
    # --- End Constraint 1 Implementation ---

    # Proceed with ONLY the best parallel cluster
    try:
        parallel_lines = all_lines_normalized[indices_best_parallel]
        # Use nanmean to be robust against potential NaNs in angles
        average_orientation = np.nanmean(line_angles[indices_best_parallel])
        if np.isnan(average_orientation):
             #print("Warning: Average orientation is NaN for the largest cluster. Skipping.")
             return
    except (TypeError, IndexError) as e:
        #print(f'Error accessing data for best parallel cluster: {e}')
        return

    # --- Line Clustering & Reduction ---
    # Transform the parallel lines by the average orientation to make them horizontal
    transformed_parallel_lines = np.zeros_like(parallel_lines)
    try:
        # Apply transform_2d safely
        for i, line in enumerate(parallel_lines):
            p1_transformed = transform_2d(line[:2], (0, 0), -average_orientation)
            p2_transformed = transform_2d(line[2:4], (0, 0), -average_orientation)
            transformed_parallel_lines[i, :2] = p1_transformed
            transformed_parallel_lines[i, 2:4] = p2_transformed
        if np.isnan(transformed_parallel_lines).any():
            ##print("Warning: NaN values found after transforming parallel lines.")
            # Handle NaN if necessary, e.g., filter them out before clustering
            transformed_parallel_lines = transformed_parallel_lines[~np.isnan(transformed_parallel_lines).any(axis=1)]
            valid_tf_lines_idx = ~np.isnan(transformed_parallel_lines).any(axis=1)
            if not np.any(valid_tf_lines_idx):
                #print("All transformed parallel lines resulted in NaN.")
                return
            transformed_parallel_lines = transformed_parallel_lines[valid_tf_lines_idx]
            if transformed_parallel_lines.shape[0] < MIN_LINES_FOR_COLINEAR_CLUSTER: # Check if enough lines remain
                ##print("Not enough valid lines remaining after transformation for collinear clustering.")
                return

    except Exception as e:
         #print(f"Error during parallel line transformation: {e}")
         return

    # Use clustering (DBSCAN on y-midpoints) to group collinear lines
    transformed_midpoints = (transformed_parallel_lines[:, :2] + transformed_parallel_lines[:, 2:4]) / 2
    transformed_midpoints_y = transformed_midpoints[:, 1].reshape(-1, 1) # Use only Y-coordinates for clustering
    # Adjust eps based on pixel scale (e.g., 10 pixels if coords are 0-W_img/H_img)
    colinear_clustering = DBSCAN(eps=COLINEAR_EPSILON, min_samples=1).fit(transformed_midpoints_y) # Clustering on Y
    labels_colinear = colinear_clustering.labels_
    unique_labels_colinear = np.unique(labels_colinear)

    transformed_reduced_lines = [] # Stores [min_x, avg_y, max_x, avg_y] in image scale (cm)
    for label_col in unique_labels_colinear:
        if label_col == -1: # Skip noise points from DBSCAN
            continue
        indices_col = np.where(labels_colinear == label_col)[0]
        if len(indices_col) == 0: # Should not happen with unique_labels, but safe check
            continue

        transformed_colinear_lines_set = transformed_parallel_lines[indices_col]

        # Ensure calculations are robust against single-line clusters if min_samples=1
        min_x = np.min(transformed_colinear_lines_set[:, [0, 2]]) # Min of all start/end x
        max_x = np.max(transformed_colinear_lines_set[:, [0, 2]]) # Max of all start/end x
        avg_y = np.nanmean((transformed_colinear_lines_set[:, 1] + transformed_colinear_lines_set[:, 3]) / 2) # Avg y of midpoints

        if np.isnan(avg_y):
            #print(f"Warning: NaN avg_y for collinear cluster {label_col}. Skipping this reduced line.")
            continue

        transformed_reduced_lines.append([min_x, avg_y, max_x, avg_y])

    # Give up if fewer than 3 lines remain
    if len(transformed_reduced_lines) < MIN_LINES_FOR_COLINEAR_CLUSTER:
        ##print(f"Found only {len(transformed_reduced_lines)} distinct collinear lines after reduction. Need at least {MIN_LINES_FOR_COLINEAR_CLUSTER}. Abandoning.")
        return
    
    # --- End Line Clustering and Reduction ---

    # --- Spacing Consistency Check

    # Convert reduced lines from image scale (cm) to meters
    transformed_reduced_lines_m = np.array(transformed_reduced_lines) * 0.01

    # Sort lines by their average Y coordinate (in transformed space)
    sort_indices = np.argsort(transformed_reduced_lines_m[:, 1])
    sorted_reduced_lines_m = transformed_reduced_lines_m[sort_indices]

    y_values_m = sorted_reduced_lines_m[:, 1] # Sorted average y-coordinates in meters

    # Calculate pairwise distances between adjacent lines
    distances_between_lines_m = np.diff(y_values_m)

    # Estimate base spacing using median of significant distances
    valid_distances = distances_between_lines_m[distances_between_lines_m > MIN_SIGNIFICANT_DISTANCE]
    if len(valid_distances) == 0:
         #print(f"No significant distances (> {MIN_SIGNIFICANT_DISTANCE}m) found between lines. Cannot check spacing. Skipping.")
         return

    estimated_base_spacing = np.median(valid_distances)
    if estimated_base_spacing <= MIN_SIGNIFICANT_DISTANCE / 2: # Check if median is reasonably large
        #print(f"Estimated base spacing ({estimated_base_spacing:.3f}m) is too small. Skipping.")
        return

    #print(f"Estimated base spacing: {estimated_base_spacing:.3f} m")

    spacing_consistent = True
    detected_multiples = []
    for d in distances_between_lines_m:
        if d <= MIN_SIGNIFICANT_DISTANCE / 2: # Ignore very small or zero gaps
            #print(f"Warning: Ignoring small gap ({d:.3f}m) for spacing check.")
            continue

        multiple = d / estimated_base_spacing
        nearest_multiple = round(multiple)

        if nearest_multiple == 0: nearest_multiple = 1 # Avoid issues if d << estimated_base_spacing

        relative_error = abs(multiple - nearest_multiple)

        # Check if relative error is within tolerance
        if relative_error > SPACING_TOLERANCE_RATIO:
            #print(f"Inconsistent spacing: Dist {d:.3f}m is not close to multiple {nearest_multiple} of base {estimated_base_spacing:.3f}m (Rel Error: {relative_error:.2f} > {SPACING_TOLERANCE_RATIO})")
            spacing_consistent = False
            break
        detected_multiples.append(nearest_multiple)

    if not spacing_consistent:
        #print("Failed spacing consistency check. Abandoning instance.")
        return
#    else:
        #print(f"Spacing consistency passed. Detected multiples (approx): {detected_multiples}")
    # --- End Constraint 3 Implementation ---

    # --- Calculations and Visualization ---
    # Calculate final metrics using the sorted, validated lines
    length = np.max(np.abs(sorted_reduced_lines_m[:, 2] - sorted_reduced_lines_m[:, 0]))
    width = y_values_m[-1] - y_values_m[0] # Total width span

    # Calculate origin in world coordinates
    # Origin based on the minimum point of the bounding box in the TRANSFORMED space
    transformed_origin_m = (np.min(sorted_reduced_lines_m[:, 0]), np.min(sorted_reduced_lines_m[:, 1]))
    # Rotate this origin back
    origin_rotated_back = transform_2d(transformed_origin_m, (0, 0), average_orientation)
    # Add the minimum offset of the INLIER points in world space
    origin_world = (origin_rotated_back[0] + min_xyz_inliers[0], origin_rotated_back[1] + min_xyz_inliers[1])

    #print(f"Detected object: Length={length:.3f}m, Width={width:.3f}m, Orientation={np.degrees(average_orientation):.1f}deg, Origin={origin_world}, Height={best_d:.3f}m")

    # --- Publish Visualization ---
    # 1. Draw lines on tops_2d image (using original image-scale coordinates)
    tops_2d_color = cv2.cvtColor(tops_2d, cv2.COLOR_GRAY2BGR) # Draw on color version
    for line in transformed_reduced_lines: # Use the cm-scale lines here
        p1 = (int(line[0]), int(line[1]))
        p2 = (int(line[2]), int(line[3]))
        cv2.line(tops_2d_color, p1, p2, color=(0, 255, 0), thickness=2) # Draw reduced lines in green


    # 2. Publish lines as ROS markers
    marker = Marker()
    marker.header.frame_id = "base_footprint"  # Ensure this matches your world frame
    marker.header.stamp = rospy.Time.now() # Use current time or provided timestamp?
    marker.ns = "detected_lines"
    marker.id = 0
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.02  # Line width in meters

    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0  # Fully opaque

    # Convert reduced lines back to world coordinates
    final_lines_world = np.zeros_like(sorted_reduced_lines_m)
    detected_z = best_d # Use the Z height of the plane where inliers were found

    for i, line in enumerate(sorted_reduced_lines_m):
         # Rotate start/end points back from transformed space
         p1_tf = line[:2]
         p2_tf = line[2:4]
         p1_rot = transform_2d(p1_tf, (0, 0), average_orientation)
         p2_rot = transform_2d(p2_tf, (0, 0), average_orientation)

         # Add world offset (min_xyz of inliers)
         p1_world = Point(x=p1_rot[0] + min_xyz_inliers[0], y=p1_rot[1] + min_xyz_inliers[1], z=detected_z)
         p2_world = Point(x=p2_rot[0] + min_xyz_inliers[0], y=p2_rot[1] + min_xyz_inliers[1], z=detected_z)

         final_lines_world[i, :2] = p1_world.x, p1_world.y
         final_lines_world[i, 2:4] = p2_world.x, p2_world.y

         marker.points.append(p1_world)
         marker.points.append(p2_world)

    try:
        pub2.publish(marker)
    except Exception as e:
        print(f"Error publishing marker: {e}")

    # 3. Publish HorizontalObservation message
    observation = HorizontalObservation()
    observation.header.stamp = time_stamp
    observation.header.frame_id = "base_footprint"  # Ensure this matches your world frame
    observation.orientation = average_orientation
    observation.length = length
    observation.spacing = estimated_base_spacing
    observation.height = best_d
    observation.lines = list(final_lines_world.flatten())
    pub.publish(observation)

    end_time = time.time()
    #print(f"Detection time: {end_time - start_time:.4f} seconds")

def depth_callback(msg):
    """Callback function for Depth images."""
    try:
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        time_stamp = msg.header.stamp
        detect(depth_image, time_stamp)
    except Exception as e:
        rospy.logerr(f"Error in Depth callback: {e}")

if __name__ == "__main__":
    rospy.init_node("horizontal_detector_node")
    bridge = CvBridge()
    video_topic = "/camera/rgb/image_raw"
    depth_topic = "/camera/depth/image"
#    rospy.Subscriber(video_topic, Image, rgb_callback, queue_size=2)
    rospy.Subscriber(depth_topic, Image, depth_callback, queue_size=2)
    pub = rospy.Publisher('/horiz_roi', HorizontalObservation, queue_size=2)
    pub2 = rospy.Publisher('/lines', Marker, queue_size=2)
    pub3 = rospy.Publisher('/cavity_detection/depth_points_2d', Image, queue_size=2)

    rospy.spin()