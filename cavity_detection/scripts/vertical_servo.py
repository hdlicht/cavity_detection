#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import scipy
from scipy.ndimage import maximum_filter1d


def rgb_callback(msg):
    global bridge
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    blurred_hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
    lower_bg = np.array([20, 40, 0])
    upper_bg = np.array([150, 255, 150])
    mask = cv2.inRange(hsv, lower_bg, upper_bg)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    inverted_mask = cv2.bitwise_not(mask_dilated)

    blank = np.zeros(cv_image.shape, dtype=np.uint8)
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    gray = cv2.equalizeHist(blurred_image, None)
    edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edges = cv2.GaussianBlur(edges, (5, 5), 0)

    # Parameters (tune based on observed image scale)
    positive_width = 3
    gap = 10
    negative_width = 3

    # Create kernel
    kernel = np.concatenate([
        -np.ones(positive_width),         # +1s for the leading edge
        np.zeros(gap),                          # gap in the middle
        np.ones(negative_width)         # -1s for the trailing edge
    ])
    #kernel = kernel - np.mean(kernel)  # zero-mean for better response

    # Example edge profile
    edge_profile = np.sum(edges, axis=0)  # 1D horizontal sum of vertical edges
    profile_image = ((edge_profile-np.min(edge_profile))/(np.max(edge_profile)-np.min(edge_profile)) * 255).astype(np.uint8)
    profile_image = np.tile(profile_image, (50, 1))
    mask_pub.publish(bridge.cv2_to_imgmsg(profile_image))

    # Convolve
    response = scipy.signal.convolve(edge_profile, kernel, mode='same')
    # Suppress everything below threshold
    threshold = np.max(response) * 0.55
    response[response < threshold] = 0

    # Non-maximum suppression (local maxima in a window)
    nms_window = 20  # larger than dash spacing
    max_response = maximum_filter1d(response, size=nms_window)
    stud_locations = np.where(response == max_response)[0]
    stud_locations = stud_locations[response[stud_locations] > 0] 

    thresh = np.mean(np.abs(edge_profile)) * 3
    pos_lines = np.where(edge_profile > thresh)[0]
    neg_lines = np.where(edge_profile < -thresh)[0]

    # Separate positive and negative edges (preserve shape)
    pos_edges = np.where(edges > 0, edges, 0)
    neg_edges = np.where(edges < 0, -edges, 0)
    # Normalize the edges
    pos_edges = cv2.normalize(pos_edges, None, 0, 255, cv2.NORM_MINMAX)
    neg_edges = cv2.normalize(neg_edges, None, 0, 255, cv2.NORM_MINMAX)
    # Convert to uint8
    pos_edges = pos_edges.astype(np.uint8)
    neg_edges = neg_edges.astype(np.uint8)
    # Threshold using Otsu
    _, pos_thresh = cv2.threshold(pos_edges, 30, 255, cv2.THRESH_BINARY)
    _, neg_thresh = cv2.threshold(neg_edges, 30, 255, cv2.THRESH_BINARY)

    # Visualize: red = positive, green = negative
    blank[:, :, 2] = pos_thresh  # Red channel
    blank[:, :, 1] = neg_thresh  # Green channel

    # turn grey image back to rgb
    #gray_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

    for line in pos_lines:
        cv2.line(blank, (line, 0), (line, blank.shape[0]), (255, 0, 0), 1)
    for line in neg_lines:
        cv2.line(blank, (line, 0), (line, blank.shape[0]), (255, 0, 255), 1)
    for line in stud_locations:
        cv2.line(cv_image, (line, 0), (line, blank.shape[0]), (0, 255, 255), 3)

    # Publish result
    image_pub.publish(bridge.cv2_to_imgmsg(cv_image, encoding="bgr8"))


if __name__ == "__main__":
    rospy.init_node("horizontal_detector_node")
    bridge = CvBridge()
    video_topic = "/camera/rgb/image_raw"
    depth_topic = "/camera/depth/image"
#    rospy.Subscriber(video_topic, Image, rgb_callback, queue_size=2)
    rospy.Subscriber(video_topic, Image, rgb_callback, queue_size=1)
    mask_pub = rospy.Publisher('/cavity_detection/mask', Image, queue_size=2)
    image_pub = rospy.Publisher('/cavity_detection/edges', Image, queue_size=2)

    rospy.spin()