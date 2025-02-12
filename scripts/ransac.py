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

    for _ in range(iterations):
        # 1. Randomly sample 3 points
        sample_indices = np.random.choice(len(points), 3, replace=False)
        sample_points = points[sample_indices]

        # 2. Calculate plane equation
        p1, p2, p3 = sample_points
        normal_vector = np.cross(p2 - p1, p3 - p1)
        a, b, c = normal_vector
        d = -np.dot(normal_vector, p1)
        
        # 3. Calculate distances from all points to the plane
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)

        # 4. Determine inliers
        inlier_indices = np.where(distances <= threshold)[0]
        inliers = points[inlier_indices]

        # 5. Update best plane if necessary
        num_inliers = len(inliers)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inliers = inliers
            best_plane = np.array([a, b, c, d])

    return best_plane, best_inliers