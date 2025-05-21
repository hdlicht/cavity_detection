#include <ros/ros.h>
#include "geometry_msgs/Pose.h"
#include "cavity_detection_msgs/Roi.h"
#include "cavity_detection_api/api.h"
#include <string>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "cavity_client");

    cavity_detection_msgs::Roi roi;
    getNearestRoi(roi);

    geometry_msgs::Pose roi_pose;
    std::string roi_id = roi.id;
    roi_pose.position.x = 4.0;
    roi_pose.position.y = 5.0;
    roi_pose.position.z = 6.0;

    updateRoi(roi_id, roi_pose, 10.0, 5.0, 2.0);

    std::string cavity_id;
    addCavity(roi_id, 2.0, 3.0, cavity_id);

    updateCavity(roi_id, cavity_id, 2.0, 3.0, 0);

    return 0;
}
