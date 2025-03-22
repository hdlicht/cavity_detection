#ifndef CAVITY_API_H
#define CAVITY_API_H

#include <geometry_msgs/Pose.h>

bool getNearestCavity(const geometry_msgs::Pose robot_pose, std::string& cavity_id, geometry_msgs::Pose& cavity_pose);
bool updateRoi(const std::string roi_id, const geometry_msgs::Pose roi_pose, const double length, const double width, const double depth);
bool addCavity(const std::string roi_id, const double y_offset, const double width, std::string& cavity_id);
bool updateCavity(const std::string roi_id, const std::string cavity_id, const double y_offset, const double width, const int status);

#endif 
