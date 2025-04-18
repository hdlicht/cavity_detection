#!/usr/bin/env python3
import rospy
import open3d as o3d
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
from scipy.spatial.transform import Rotation as R

pub = None

def plane_to_pose(plane_model):
    a, b, c, d = plane_model

    # Normalize the normal vector
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    # Position: Choose a point on the plane (z=0, y=0 if possible, solve for x)
    # If a != 0, we can find x directly:
    if abs(a) > 1e-6:
        position = np.array([-d / a, 0, 0])
    elif abs(b) > 1e-6:
        position = np.array([0, -d / b, 0])
    else:
        position = np.array([0, 0, -d / c])

    # Compute rotation matrix to align z-axis with the plane normal
    z_axis = normal  # Plane normal
    x_axis = np.array([1, 0, 0])
    if np.allclose(z_axis, x_axis):  # Avoid singularity if normal is along x-axis
        x_axis = np.array([0, 1, 0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)

    # Rotation matrix
    rotation_matrix = np.column_stack((x_axis, y_axis, normal))

    # Convert to quaternion
    quaternion = R.from_matrix(rotation_matrix).as_quat()
    return position, quaternion


def rect_callback(msg):
    data = np.array(msg.data).reshape(-1, 6)
    for i in range(data.shape[0]):
        plane_model = data[i, :4]
        size = data[i, 4:6]
        create_plane_marker(plane_model, size, marker_id=i)
    


def create_plane_marker(plane_model, size, marker_id=0):
    """
    Creates a visualization marker representing a plane.

    Parameters:
    - plane_model: [a, b, c, d] plane equation coefficients
    - center: Center of the plane [x, y, z]
    - size: Plane dimensions (width, height)
    - marker_id: Unique ID for RViz marker
    """
    global pub
    if pub is None:
        pub = rospy.Publisher("plane_marker", Marker, queue_size=10)
    marker = Marker()
    marker.header.frame_id = "map"  # Change frame_id as needed (e.g., "base_link")
    marker.header.stamp = rospy.Time.now()
    marker.ns = "plane_marker"
    marker.id = marker_id
    marker.type = Marker.CUBE  # Represent the plane as a cube
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration(5)  # Marker lasts for 5 seconds

    # Set orientation
    position, quaternion = plane_to_pose(plane_model)
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    marker.pose.orientation.x = quaternion[0]
    marker.pose.orientation.y = quaternion[1]
    marker.pose.orientation.z = quaternion[2]
    marker.pose.orientation.w = quaternion[3]

    # Set dimensions (width, height, very thin thickness)
    marker.scale.x = size[0]
    marker.scale.y = size[1]
    marker.scale.z = 0.01  # Thin plane

    # Set color
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 0.7  # Transparency

    pub.publish(marker)

def main():
    rospy.init_node("plane_visualization_node")
    
    rospy.Subscriber('/detected_shapes', Float32MultiArray, rect_callback, queue_size=10)

    rospy.spin()

if __name__ == "__main__":
    main()
