import numpy as np
import math
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist

def transform_2d(points, origin=(0,0), angle=0):
    T = np.array([[np.cos(angle), -np.sin(angle), origin[0]], 
                [np.sin(angle), np.cos(angle), origin[1]], 
                [0, 0, 1]])
    points = np.hstack((points, np.ones((points.shape[0], 1))))  # Convert to homogeneous coordinates
    new_points = T @ points.T  # Apply transformation
    new_points = new_points.T[:, :2]  # Convert back to Cartesian coordinates
    return new_points

def transform_3d(points, trans, quat):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(quat).as_matrix()  # Rotation matrix from quaternion
    T[:3, 3] = trans
    points = np.hstack((points, np.ones((points.shape[0], 1))))  # Convert to homogeneous coordinates
    new_points = T @ points.T  # Apply transformation
    new_points = new_points.T[:, :3]  # Convert back to Cartesian coordinates
    return new_points

def invert_2d_transform(T):
    R = T[:2, :2]  # Rotation part
    t = T[:2, 2]   # Translation part

    R_inv = R.T  # Transpose for inverse rotation
    t_inv = -np.dot(R_inv, t)  # Transform translation

    # Construct the inverse homogeneous transform
    T_inv = np.eye(3)
    T_inv[:2, :2] = R_inv
    T_inv[:2, 2] = t_inv

    return T_inv

# Measurement 1: Intrinsics [theta, L, s]
# State x: [px, py, theta, L, s]
def h1_intrinsics(x):
    return np.array([x[2], x[3], x[4], x[5]])

H1_intrinsics_jacobian = np.array([
    [0., 0., 1., 0., 0., 0.],
    [0., 0., 0., 1., 0., 0.],
    [0., 0., 0., 0., 1., 0.], 
    [0., 0., 0., 0., 0., 1.]
])

# Measurement 2: Position [px, py] (derived from offset)
# State x: [px, py, theta, L, s]
def h2_position(x):
    return np.array([x[0], x[1]])

H2_position_jacobian = np.array([
    [1., 0., 0., 0., 0., 0.],
    [0., 1., 0., 0., 0., 0.]
])

# --- Custom residual function for angle wrapping (for update 1) ---
def residual_with_angle(a, b):
    """ Custom residual function: z - h(x) that handles angle wrapping.
        Assumes angle (theta) is at index 2 in both a (measurement) and b (h(x)).
    """
    y = a - b
    # Wrap angle difference (element at index 2, assuming z1=[theta, L, s])
    y[0] = math.atan2(math.sin(y[0]), math.cos(y[0])) # <--- Adjust index if z1 ordering changes
    return y

def mutual_nearest_neighbors(set1, set2):
    # Compute pairwise distances
    dist_matrix = cdist(set1, set2)

    # Nearest neighbor indices
    nn_1_to_2 = np.argmin(dist_matrix, axis=1)  # Closest point in set2 for each point in set1
    nn_2_to_1 = np.argmin(dist_matrix, axis=0)  # Closest point in set1 for each point in set2
    mutual_matches = []
    # Find mutual matches
    for i, j in enumerate(nn_1_to_2): 
        if nn_2_to_1[j] == i:
            mutual_matches.append([i, j])
    # Convert to numpy arrays
    mutual_matches = np.array(mutual_matches)
    # Calculate offsets
    offsets_xy = set2[mutual_matches[:, 1]] - set1[mutual_matches[:, 0]]
    # Get list of indices for no match
    no_match = np.setdiff1d(np.arange(len(set2)), mutual_matches[:, 1])

    return mutual_matches, offsets_xy, no_match

def points_from_grid(grid_msg):
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
    return np.array([[x * resolution + origin_x, y * resolution + origin_y] for y, x in occupied_cells])

def scan_to_xy(scan_msg):
    """Convert LaserScan to Nx2 numpy array of 2D points in laser frame."""
    ranges = np.array(scan_msg.ranges)
    angle_min = scan_msg.angle_min
    angle_increment = scan_msg.angle_increment

    # Create angle array
    angles = angle_min + np.arange(len(ranges)) * angle_increment

    # Filter out invalid ranges
    valid_mask = (
        np.isfinite(ranges) &
        (ranges >= scan_msg.range_min) &
        (ranges <= scan_msg.range_max)
    )

    ranges = ranges[valid_mask]
    angles = angles[valid_mask]

    xs = ranges * np.cos(angles)
    ys = ranges * np.sin(angles)

    return np.stack((xs, ys), axis=-1)  # Shape: (N, 2)    