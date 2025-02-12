# Cavity Detection Module

## Overview

The Cavity Detection Module is a ROS-based package designed to detect and map cavities using RGB-D images. The module processes images to identify regions of interest (ROIs) and publishes these ROIs for further analysis and visualization. The package includes several scripts for detecting horizontal and vertical ROIs, processing observations, and maintaining a map of detected cavities.


## Scripts

### `scripts/horizontal_roi.py`

This script initializes the ROS node `horizontal_detector_node` and processes RGB-D images to detect horizontal ROIs. It uses the RANSAC algorithm for plane fitting and publishes the detected ROIs to the `/horiz_roi` topic.

### `scripts/vertical_roi.py`

This script initializes the ROS node `vertical_detector_node` and processes RGB-D images to detect vertical ROIs. It uses the RANSAC algorithm for plane fitting and publishes the detected ROIs to the `/vert_roi` topic.

### `scripts/cavity_map.py`

This script initializes the ROS node `cavity_map` and subscribes to the `/horiz_roi` and `/vert_roi` topics to receive horizontal and vertical ROIs, respectively. It maintains a list of detected cavities and publishes markers for visualization.

### `scripts/ransac.py`

This script contains the implementation of the RANSAC algorithm for plane fitting. It is used by both `scripts/horizontal_roi.py` and `scripts/vertical_roi.py` to fit planes to point clouds.

## Messages

### `Roi.msg`

Defines the structure of a Region of Interest (ROI) message, including fields for the center, orientation, dimensions, and type of the ROI.

## Launch Files

### `launch/roi.launch`

Launches the `vertical_roi`, `horizontal_roi`, and `cavity_map` nodes for detecting and mapping cavities.

## Dependencies

- ROS
- NumPy
- OpenCV
- tf
- tf2_ros
- cv_bridge
- visualization_msgs
- geometry_msgs
- sensor_msgs
- std_msgs