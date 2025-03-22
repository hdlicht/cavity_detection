#!/usr/bin/env python3

import numpy as np
from cavity_detection_msgs.msg import Roi
import tf.transformations
import tf
import uuid

class Cavity:
    def __init__(self, id, parent, front, orientation, length, width, depth):
        self.id = id
        self.parent = parent
        self.front = front
        self.orientation = orientation
        self.length = length
        self.width = width
        self.depth = depth
        self.filled = False

class RegionOfInterest:
    def __init__(self, id, roi_type, first_observation):
        self.type = roi_type
        self.id = id
        self.origin = first_observation.origin
        self.orientation = first_observation.orientation
        self.length = first_observation.length
        self.width = first_observation.width
        self.depth = first_observation.depth
        self.num_cavities = first_observation.num_cavities
        self.cavity_width = first_observation.cavity_width
        self.observations = []
        self.observations.append(first_observation)
        self.cavities = {}

    def add_observation(self, observation):
        count = len(self.observations)
        self.observations.append(observation)
        if count > 20:
            self.observations.pop(0)
        self.origin = (self.origin * count + observation.origin) / (count + 1)
        self.orientation = (self.orientation * count + observation.orientation) / (count + 1)
        self.orientation = self.orientation / np.linalg.norm(self.orientation)
        self.length = (self.length * count + observation.length) / (count + 1)
        self.width = (self.width * count + observation.width) / (count + 1)
        self.depth = (self.depth * count + observation.depth) / (count + 1)
        self.num_cavities = max(self.num_cavities, observation.num_cavities)
        observed_width = observation.cavity_width
        expected_width = self.width / self.num_cavities
        expanded_width = self.width / (self.num_cavities + 1)

        if np.abs(observed_width - expected_width) < 0.03:
            self.cavity_width = observed_width
        elif np.abs(observed_width - expanded_width) < 0.03:
            self.cavity_width = observed_width
            self.num_cavities += 1

    def locate_cavities(self):
        # Calculate the location of a point in frame A
        R = tf.transformations.quaternion_matrix(self.orientation)
        T = np.eye(4)
        T[:3, 3] = self.origin
        T[:3, :3] = R[:3, :3]
        for i in range(self.num_cavities):
            local_point = np.array([self.origin[0], self.origin[1] - self.cavity_width * (i + 0.5), 0])
            world_point = np.dot(T, local_point)
            front = world_point[:3]
            cavity_i = Cavity(front, self.id, self.orientation, self.length, self.width, self.depth)
            self.cavities.append(cavity_i)


class Observation:
    def __init__(self, origin, orientation, length, width, depth, observation_distance, observation_angle, num_cavities, cavity_width):
        self.origin = np.array([origin.x, origin.y, origin.z])
        self.orientation = np.array([orientation.x, orientation.y, orientation.z, orientation.w])
        self.length = length
        self.width = width
        self.depth = depth
        self.observation_distance = observation_distance
        self.observation_angle = observation_angle
        self.num_cavities = num_cavities
        self.cavity_width = cavity_width

    def is_overlapping(self, roi:Roi):

        def get_bounding_box(roi):

            # p1 = np.array([-roi.length/2, -roi.width/2, 0, 1])
            # p2 = np.array([-roi.length/2, roi.width/2, 0, 1])
            # p3 = np.array([roi.length/2, -roi.width/2, 0, 1])
            # p4 = np.array([roi.length/2, roi.width/2, 0, 1])
            p1 = np.array([0, 0, 0, 1])
            p2 = np.array([0, roi.width, 0, 1])
            p3 = np.array([roi.length, 0, 0, 1])
            p4 = np.array([roi.length, roi.width, 0, 1])
            R = tf.transformations.quaternion_matrix([roi.orientation[0], roi.orientation[1], roi.orientation[2], roi.orientation[3]])
            T = np.eye(4)
            T[:3, 3] = [roi.origin[0], roi.origin[1], roi.origin[2]]
            T[:3, :3] = R[:3, :3]
            p1 = np.dot(T, p1)[:2]
            p2 = np.dot(T, p2)[:2]
            p3 = np.dot(T, p3)[:2]
            p4 = np.dot(T, p4)[:2]
            x_coords = [p1[0], p2[0], p3[0], p4[0]]
            y_coords = [p1[1], p2[1], p3[1], p4[1]]
            return min(x_coords), max(x_coords), min(y_coords), max(y_coords)

        x1_min, x1_max, y1_min, y1_max = get_bounding_box(roi)
        x2_min, x2_max, y2_min, y2_max = get_bounding_box(self)

        return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)