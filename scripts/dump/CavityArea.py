import numpy as np 
from cavity_detection.msg import HorizRoi


class CavityArea:
    def __init__(self, area_id, ):
        """
        :param area_id: Unique identifier for the area.
        :param bounding_edges: Initial bounding edges (e.g., coordinates or dimensions).
        """
        self.area_id = area_id
        self.bounding_edges = bounding_edges  # Could be a polygon or rectangle
        self.cavities = []  # List of associated cavities
        self.observations = []  # List of observations refining the area

    def update_edges(self, new_edges):
        """Update bounding edges to refine the area."""
        self.bounding_edges = new_edges

    def add_cavity(self, cavity):
        """Add a cavity to the area."""
        self.cavities.append(cavity)

    def add_observation(self, observation):
        """Add an observation to the area."""
        self.observations.append(observation)


class Cavity:
    def __init__(self, cavity_id, location, size=None, depth=None):
        """
        :param cavity_id: Unique identifier for the cavity.
        :param location: Location within the area (e.g., coordinates).
        :param size: Size of the cavity (optional).
        :param depth: Depth of the cavity (optional).
        """
        self.cavity_id = cavity_id
        self.location = location
        self.size = size
        self.depth = depth

    def update_details(self, size, depth):
        """Update size and depth information for the cavity."""
        self.size = size
        self.depth = depth


class Observation:
    def __init__(self, observation_id, data, timestamp=None):
        """
        :param observation_id: Unique identifier for the observation.
        :param data: Image or data collected in the observation.
        :param timestamp: When the observation was collected.
        """
        self.observation_id = observation_id
        self.data = data
        self.timestamp = timestamp

    def process(self):
        """Process the observation (e.g., analyze image, extract features)."""
        pass  # Add logic to analyze the observation
