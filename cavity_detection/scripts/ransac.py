import numpy as np

def ransac_plane_fitting(points, threshold, iterations):
    """
    Fits a plane to a set of points using the RANSAC algorithm.

    Args:
        points: A numpy array of shape (n, 3) representing the point cloud.
        threshold: The distance threshold for considering a point an inlier.
        iterations: The number of RANSAC iterations.

    Returns:
        A tuple containing:
            - The equation of the plane (a, b, c, d) as a numpy array.
            - The inlier points as a numpy array.
    """
    best_inliers = None
    best_plane = None
    max_inliers = 0
    num_points = points.shape[0]

    for i in range(iterations):
        # 1. Randomly sample 3 points
        sample_indices = np.random.choice(len(points), 3, replace=False)
        sample_points = points[sample_indices]

        # 2. Calculate plane equation
        p1, p2, p3 = sample_points
        normal_vector = np.cross(p2 - p1, p3 - p1)
        normal_vector /= np.linalg.norm(normal_vector)
        a, b, c = normal_vector
        if c < 0:
            a, b, c = -a, -b, -c
        d = np.dot(normal_vector, p1)
        
        # 3. Calculate distances from all points to the plane
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)

        # 4. Determine inliers
        inlier_indices = np.where(distances <= threshold)[0]

        # 5. Update best plane if necessary
        num_inliers = len(inlier_indices)
        if num_inliers > max_inliers:
            if num_inliers > 10000:
                return np.array([a, b, c, d]), inlier_indices
            max_inliers = num_inliers
            best_inliers = inlier_indices
            best_plane = np.array([a, b, c, d])

    return best_plane, best_inliers