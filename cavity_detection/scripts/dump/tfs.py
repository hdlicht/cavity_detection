import tf
import rospy
import numpy as np

def print_transform_matrix():
    rospy.init_node('transform_listener')
    listener = tf.TransformListener()

    try:
        listener.waitForTransform('base_footprint', 'camera_rgb_optical_frame', rospy.Time(), rospy.Duration(4.0))
        (trans, rot) = listener.lookupTransform('base_footprint', 'camera_rgb_optical_frame', rospy.Time(0))

        translation_matrix = tf.transformations.translation_matrix(trans)
        rotation_matrix = tf.transformations.quaternion_matrix(rot)
        transform_matrix = np.dot(translation_matrix, rotation_matrix)

        print("Transform matrix from /camera_link to /base_footprint:")
        print(transform_matrix)

    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.logerr("Transform lookup failed")

if __name__ == '__main__':
    print_transform_matrix()