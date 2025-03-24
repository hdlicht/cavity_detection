#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import threading
import numpy as np
from sklearn.cluster import DBSCAN
import threading
from ransac import ransac_plane_fitting
from cavity_detection_msgs.msg import Roi
from geometry_msgs.msg import Point, Quaternion
from visualization_msgs.msg import Marker
import open3d as o3d
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from random import randint
from math import atan2

# Global shared buffer
data_buffer = {"rgb": None, "depth": None}
buffer_lock = threading.Lock()

# Desired frequency for fusion (Hz)
RUN_RATE = 5  
SIZE_FACTOR = 0.5
GROUND_NORMAL = np.array([0, 0, 1])  
NORMAL_THRESHOLD = 0.2

# ROS Publisher1
pub3 = None

# K_rgb = np.array([[570.342, 0.0,     314.5], 
#                 [0.0,     570.342, 235.5],
#                 [0.0,     0.0,     1.0]])

K_rgb = np.array([[910.1351318359375, 0.0, 643.9998779296875], 
                  [0.0, 910.0550537109375, 370.72381591796875], 
                  [0.0, 0.0, 1.0]])

K_depth = np.array([[910.1351318359375, 0.0, 643.9998779296875], 
                    [0.0, 910.0550537109375, 370.72381591796875], 
                    [0.0, 0.0, 1.0]])

K_scaler = np.array([[SIZE_FACTOR], [SIZE_FACTOR], [1]])
K_depth *= K_scaler
K_rgb *= K_scaler

# K_depth = np.array([[570.342, 0.0,     319.5],
#                     [0.0,     570.342, 239.5],
#                     [0.0,     0.0,     1.0]])



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
time_stamp = None
calibration_count = 0

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
    valid = zs > 0  # Filter out invalid depth values
    xs = (points_2d[:, 0] - K_rgb[0, 2]) / K_rgb[0, 0] * zs
    ys = (points_2d[:, 1] - K_rgb[1, 2]) / K_rgb[1, 1] * zs
    points_3d = np.vstack((xs, ys, zs)).T
    return points_3d[valid]

def publish_pointcloud(pcd):
    """Convert Open3D point cloud to ROS PointCloud2 and publish."""
    global pub3

    if pub3 is None:
        return

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "base_footprint"  # Adjust frame as needed

    points = np.asarray(pcd.points)
    cloud_msg = pc2.create_cloud_xyz32(header, points)

    pub3.publish(cloud_msg)

def detect(event):
    """Periodically process the fusion of RGB and Depth images."""
    global data_buffer, buffer_lock, pub, models, calibration_count, average_model
    with buffer_lock:
        if data_buffer["rgb"] is not None and data_buffer["depth"] is not None:
            rgb_image = data_buffer["rgb"]
            depth_image = data_buffer["depth"]
            rgb_image = cv2.resize(rgb_image, (0, 0), fx=SIZE_FACTOR, fy=SIZE_FACTOR)
            depth_image = cv2.resize(depth_image, (0, 0), fx=SIZE_FACTOR, fy=SIZE_FACTOR)
            depth_image = depth_image / 1000.
            depth_image = cv2.GaussianBlur(depth_image, (7, 7), 5)
            H, W = rgb_image.shape[:2]
            
            bottom = [0, H//2, W, H]
            # convert all pixels in the rectangle to 3d
            x, y = np.meshgrid(range(bottom[0], bottom[2]), range(bottom[1], bottom[3]))
            points_2d = np.vstack([x.ravel(), y.ravel()]).T
            points = get_3d_points(depth_image, points_2d)

            # Transform the points to the world frame
            points = np.hstack((points, np.ones((points.shape[0], 1))))
            points = np.dot(T_camera_world, points.T).T
            points = points[:, :3]

            min_xyz = np.nanmin(points, axis=0)
            # print(f"Min XYZ: {min_xyz}")

            if calibration_count < 10:
                # Fit a plane to the points with RANSAC
                plane_model, inliers = ransac_plane_fitting(points, threshold=0.01, iterations=100)
                models.append(plane_model)
                calibration_count += 1
                return
            
            if calibration_count == 10:
                average_model = np.array(models).mean(axis=0)
                a, b, c, d = average_model
                if np.dot(np.array([a, b, c]), GROUND_NORMAL) > 0.95:
                    print(f"Calibration done. Plane model: {a}x + {b}y + {c}z + {d} = 0")
                    calibration_count += 1
                else:
                    print("Calibration failed")
                    print(a, b, c, d)
                    calibration_count = 0
                    models = []
                    return

            a, b, c, d = average_model
            inliers = (np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) < 0.05)

            points = points[~inliers]
            # Generate Point Cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Estimate normals
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
            normals = np.asarray(pcd.normals)

            # Filter points that are parallel to the ground
            dot_products = np.abs(normals @ GROUND_NORMAL)  # Compute |dot(normal, ground_normal)|
            filtered_indices = np.where((dot_products > (1 - NORMAL_THRESHOLD)))[0]
            filtered_pcd = pcd.select_by_index(filtered_indices)
            publish_pointcloud(filtered_pcd)
            points = np.asarray(filtered_pcd.points)

            # Calculate distances of all points to the new plane
            best_inliers = []
            best_count = 0
            best_d = 0

            # find the plane parralel to the ground with the most inliers
            for delta_d in np.linspace(.1, .6, 20):
                new_d = delta_d
                distances = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] - new_d)
                # Identify inliers based on the distance threshold
                inlier_indices = np.where(distances < 0.0127)[0]
                if len(inlier_indices) > best_count:
                    best_count = len(inlier_indices)
                    best_inliers = inlier_indices
                    best_d = delta_d

            # print(f"Best d: {best_d}")
            if len(best_inliers) < 10:
                return
            
            xy = points[best_inliers]
            xy = xy - min_xyz
            W = int(max(xy[:, 0])*100)
            H = int(max(xy[:, 1])*100)

            # Show the inliers on a blank image
            blank = np.zeros((H, W), dtype=np.uint8)
            for x, y, _ in xy:
                blank[int(y*100)-1, int(x*100)-1] = 255
            blank = cv2.dilate(blank, None, iterations=2)
            blank = cv2.erode(blank, None, iterations=1)

            # Find lines in the image
            lines = cv2.HoughLinesP(blank, 1, np.pi/180, 100, minLineLength=70, maxLineGap=20)
            if lines is None:
                return
            lines = lines.flatten().reshape(-1, 4)
            new_lines = normalize_lines(lines)
            slopes = (new_lines[:, 3] - new_lines[:, 1]) / (new_lines[:, 2] - new_lines[:, 0])
            angles = [atan2(line[3] - line[1], line[2] - line[0]) for line in new_lines]
            angles = np.array(angles)
            # cluster the new lines based on the slope
            clustering = DBSCAN(eps=.1, min_samples=3).fit(angles.reshape(-1, 1))
            labels = clustering.labels_
            unique_labels = np.unique(labels)
            orientation = 0

            for i, label in enumerate(unique_labels):
                color = (randint(0, 255), randint(0, 255), randint(0, 255))
                if label == -1:
                    continue
                indices = np.where(labels == label)[0]
                if len(indices) < 3:
                    continue
                print(indices)
                try:
                    lines = new_lines[indices]
                except TypeError:
                    print('type thing')
                    continue
                average_slope = np.mean(slopes[indices])
                p1 = lines[:,:2]
                # Use clustering to group collinear lines and reduce output
                clustering = DBSCAN(eps=20, min_samples=1).fit(p1)
                labels = clustering.labels_
                unique_labels = np.unique(labels)
                blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)
                new_lines = []
                for i, label in enumerate(unique_labels):
                    if label == -1:
                        continue
                    indices = np.where(labels == label)[0]
                    cluster_lines = lines[indices]
                    min_x = np.min(cluster_lines[:, 0])
                    max_x = np.max(cluster_lines[:, 2])
                    min_y = np.min(cluster_lines[:, 1])
                    max_y = np.max(cluster_lines[:, 3])
                    new_lines.append([min_x, min_y, max_x, max_y])
                if len(new_lines) == 0:
                    return
                cluster_lines = np.array(new_lines)

                
                # sort the lines by x1
                cluster_lines = cluster_lines[np.argsort(cluster_lines[:, 0])]
                # find midpoints of first line
                midpoints = (cluster_lines[:, :2] + cluster_lines[:, 2:]) / 2
                # draw a line that is perpendicular to the slope of the lines and passes through the midpoint
                perp_slope = -1 / average_slope
                # calculate the y-intercept
                intercept = midpoints[0, 1] - perp_slope * midpoints[0, 0]
                # calculate intersections of the perpendicular line with each clusterline
                intersections = np.zeros((len(cluster_lines), 2))
                for j, line in enumerate(cluster_lines):
                    x1, y1, x2, y2 = line
                    m = (y2 - y1) / (x2 - x1)
                    b = y1 - m * x1
                    x = (b - intercept) / (perp_slope - m)
                    y = perp_slope * x + intercept
                    intersections[j] = [x, y]
                    print(x, y)
                    cv2.circle(blank, (int(x), int(y)), 5, (0, 0, 255), -1)

                # calculate the perpendicular distance between adjacent lines
                distances = np.sqrt(np.sum((intersections[1:] - intersections[:-1])**2, axis=1))
                if len(distances)==0:
                    continue
                if max(distances) < 0.1:
                    continue

                for line in cluster_lines:
                    p1 = (line[0], line[1])
                    p2 = (line[2], line[3])
                    cv2.line(blank, p1, p2, color=color, thickness=2)
                    cv2.circle(blank, p1, 5, (255, 255, 0), -1)
                    cv2.putText(blank, f"({p1[0]},{p1[1]})", (p1[0], p1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


                # draw a rectangle around the cluster
                pts=[cluster_lines[0,:2], 
                     cluster_lines[0,2:], 
                     cluster_lines[-1, 2:], 
                     cluster_lines[-1,:2]]

                pts = np.array(pts, np.int32)
                pts = pts.reshape((-1, 1, 2))

                # convert to real world coordinates

                cluster_lines = cluster_lines * 0.01
                distances = distances * 0.01
                offset = np.array([min_xyz[0], min_xyz[1], min_xyz[0], min_xyz[1]]).reshape(1, 4)
                cluster_lines = cluster_lines + offset
                lengths = np.sqrt(np.sum((cluster_lines[:, :2] - cluster_lines[:, 2:])**2, axis=1))
                orientation = np.arctan(average_slope)              
                center = np.mean(cluster_lines.reshape(-1,2), axis=0)

                # publish lines as markers for debugging

                marker = Marker()
                marker.header.frame_id = "base_footprint"  # Change to your frame
                marker.header.stamp = rospy.Time.now()
                marker.ns = "lines"
                marker.id = 0
                marker.type = Marker.LINE_LIST
                marker.action = Marker.ADD        
                marker.scale.x = 0.02  # Line width

                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0  # Fully opaque

                for line in cluster_lines:
                    p1 = Point(x=line[0], y=line[1], z=0.5)
                    p2 = Point(x=line[2], y=line[3], z=0.5)
                    marker.points.append(p1)
                    marker.points.append(p2)
                
                pub2.publish(marker)

                msg = Roi()
                msg.header.frame_id = "base_footprint"
                msg.header.stamp = time_stamp
                msg.roi_type = 0
                msg.center = Point(center[0], center[1], 0)
                msg.orientation = Quaternion(x=0, y=0, z=np.sin(orientation/2), w=np.cos(orientation/2))
                msg.length = np.max(lengths)
                msg.width = np.sum(distances)
                msg.depth = best_d
                msg.num_cavities = len(distances)
                msg.cavity_width = np.median(distances)

                pub.publish(msg)

            cv2.putText(blank, f"{(orientation * 180 / np.pi):.2f}", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow("Inliers", blank)
            cv2.waitKey(1)

        else:
            rospy.logwarn("Waiting for both RGB and Depth images to be received.")

def rgb_callback(msg):
    """Callback function for RGB images."""
    try:
        rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with buffer_lock:
            data_buffer["rgb"] = rgb_image
    except Exception as e:
        rospy.logerr(f"Error in RGB callback: {e}")

def depth_callback(msg):
    """Callback function for Depth images."""
    global time_stamp
    try:
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with buffer_lock:
            data_buffer["depth"] = depth_image
            time_stamp = msg.header.stamp
    except Exception as e:
        rospy.logerr(f"Error in Depth callback: {e}")

if __name__ == "__main__":
    rospy.init_node("horizontal_detector_node")
    bridge = CvBridge()
    
    # Subscribe to RGB and Depth topics
    # video_topic = "/camera/rgb/image_raw"
    # depth_topic = "/camera/depth/image"
    video_topic = "/camera/color/image_raw"
    depth_topic = "/camera/aligned_depth_to_color/image_raw"
    rospy.Subscriber(video_topic, Image, rgb_callback, queue_size=2)
    rospy.Subscriber(depth_topic, Image, depth_callback, queue_size=2)
    pub = rospy.Publisher('/horiz_roi', Roi, queue_size=2)
    pub2 = rospy.Publisher('/lines', Marker, queue_size=2)
    pub3 = rospy.Publisher("/filtered_pointcloud", PointCloud2, queue_size=1)



    # Timer to call process_fusion() periodically (e.g., every 100ms)
    rospy.Timer(rospy.Duration(1.0 / RUN_RATE), detect)
    rospy.spin()