#ifndef API_H
#define API_H

#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <cavity_detection_msgs/GetNearestCavity.h>
#include <cavity_detection_msgs/UpdateRoi.h>
#include <cavity_detection_msgs/AddCavity.h>
#include <cavity_detection_msgs/UpdateCavity.h>
#include <string>

// Function to get the nearest cavity based on the robot's position
bool getNearestCavity(const geometry_msgs::Pose robot_pose, std::string& cavity_id, geometry_msgs::Pose& cavity_pose);

// Function to update the region of interest (ROI)
bool updateRoi(const std::string roi_id, const geometry_msgs::Pose roi_pose, const double length, const double width, const double depth);

// Function to add a new cavity
bool addCavity(const std::string roi_id, const double y_offset, const double width, std::string& cavity_id);

// Function to update an existing cavity
bool updateCavity(const std::string roi_id, const std::string cavity_id, const double y_offset, const double width, const int status);

#endif // API_H
