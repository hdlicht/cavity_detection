import numpy as np
from nav_msgs.msg import OccupancyGrid
from scipy.ndimage import convolve, binary_dilation
from sklearn.cluster import DBSCAN
import rospy

class MapWrapper:
    def __init__(self, occupancy_grid):
        self.data = np.array(occupancy_grid.data).reshape(
            occupancy_grid.info.height, occupancy_grid.info.width
        )
        self.frontier_mask = np.zeros_like(self.data)
        self.info = occupancy_grid.info
        self.resolution = self.info.resolution
        self.origin = self.info.origin.position
        self.frontier_mask_pub = rospy.Publisher(
            "/frontier_mask",
            OccupancyGrid,
            queue_size=1,
        )

    def update(self, occupancy_grid):
        self.data = np.array(occupancy_grid.data).reshape(
            occupancy_grid.info.height, occupancy_grid.info.width
        )

    def grid_to_world(self, x, y):
        wx = self.origin.x + (x + 0.5) * self.resolution
        wy = self.origin.y + (y + 0.5) * self.resolution
        return wx, wy

    def world_to_grid(self, wx, wy):
        gx = int((wx - self.origin.x) / self.resolution)
        gy = int((wy - self.origin.y) / self.resolution)
        return gx, gy

    def is_free(self, x, y):
        return self.data[y, x] == 0

    def is_occupied(self, x, y):
        return self.data[y, x] == 100

    def is_unknown(self, x, y):
        return self.data[y, x] == -1
    
    def get_frontiers(self):
        free_mask = self.data == 0
        unknown_mask = self.data == -1
        occupied_mask = self.data == 100

        neighbor_kernel = np.array([[1, 1, 1],
                                   [1, 0, 1],
                                   [1, 1, 1]])
        
        # Find frontiers by checking for free cells adjacent to unknown cells
        frontier_neighbors = convolve(unknown_mask.astype(int), neighbor_kernel, mode='constant', cval=0)
        frontier_mask = (frontier_neighbors > 0) & free_mask
        # Publish the frontier mask

        frontier_cells = np.argwhere(frontier_mask)
        if frontier_cells.size == 0:
            rospy.loginfo("No frontiers found.")
            return
        # Use DBSCAN to cluster frontiers
        clustering = DBSCAN(eps=2, min_samples=4).fit(frontier_cells)
        labels = clustering.labels_
        unique_labels = set(labels)
        num = len(unique_labels)
        inc = 100 // num
        for label in unique_labels:
            if label == -1:
                continue
            cluster_cells = frontier_cells[labels == label]
            for cell in cluster_cells:
                self.frontier_mask[cell[0], cell[1]] = inc * (label+1)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        if not unique_labels:
            rospy.loginfo("No clusters found.")
            return
        # Create a new occupancy grid message for the frontier mask
        frontier_mask_msg = OccupancyGrid()
        frontier_mask_msg.header.stamp = rospy.Time.now()
        frontier_mask_msg.header.frame_id = "map"
        frontier_mask_msg.info = self.info
        frontier_mask_msg.data = self.frontier_mask.flatten().tolist()
        self.frontier_mask_pub.publish(frontier_mask_msg)
        # find the size and centroid of each cluster
        cluster_sizes = [np.sum(labels == label) for label in unique_labels]
        cluster_centroid_grid = [np.mean(frontier_cells[labels == label], axis=0) for label in unique_labels]
        cluster_centroids = [self.grid_to_world(centroid[1], centroid[0]) for centroid in cluster_centroid_grid]
        return cluster_centroids, cluster_sizes
