#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Int16MultiArray.h>
#include <std_msgs/Header.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <cavity_detection_msgs/Roi.h>
#include <cavity_detection_msgs/LogoObservation.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <tf/transform_listener.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "logo_detector/detector.h"  // Include your detector
#include <ros/package.h>

std::string base_path = ros::package::getPath("cavity_detection");
std::string param_path = base_path + "/model/nanodet.param";
std::string bin_path = base_path + "/model/nanodet.bin";

// Global shared buffer
cv::Mat rgb_image;
ros::Time time_stamp;
std::mutex buffer_mutex;
ros::Publisher vert_pub, image_pub;
tf::TransformListener *tf_listener;

// Initialize NanoDet detector
NanoDet detector(param_path.c_str(), bin_path.c_str(), false);

// Desired frequency (Hz)
const double RUN_RATE = 5.0;
                            
void rgb_callback(const sensor_msgs::ImageConstPtr &msg)
{
    try
    {
        time_stamp = msg->header.stamp;
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        std::lock_guard<std::mutex> lock(buffer_mutex);
        rgb_image = cv_ptr->image.clone();
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("Error in RGB callback: %s", e.what());
    }
}

bool compare_boxes(const BoxInfo &a, const BoxInfo &b) {
    return a.x1 < b.x1;
}

// Merge overlapping boxes
std::vector<BoxInfo> merge_boxes(std::vector<BoxInfo> &detections) {
    if (detections.empty()) return {};

    // Sort detections by x1
    std::sort(detections.begin(), detections.end(), compare_boxes);

    std::vector<BoxInfo> merged;
    BoxInfo current_box = detections[0];

    for (size_t i = 1; i < detections.size(); ++i) {
        BoxInfo &box = detections[i];
        float box_width = box.x2 - box.x1;
        // If overlapping in x direction, update bounding box
        if (box.x1 - box_width <= current_box.x2) {
            current_box.x2 = std::max(current_box.x2, box.x2);
            current_box.y1 = std::min(current_box.y1, box.y1);
            current_box.y2 = std::max(current_box.y2, box.y2);
        } else {
            merged.push_back(current_box);
            current_box = box;
        }
    }
    merged.push_back(current_box);
    return merged;
}

std::vector<BoxInfo> detection_pipeline(const cv::Mat &image)
{
    int width = image.cols;
    int height = image.rows;
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(320, 320));
    std::vector<BoxInfo> detections = detector.detect(resized_image, 0.5, 0.4);
    if (detections.size() < 3)
        return {};
    // Merge overlapping boxes
    std::vector<BoxInfo> merged_boxes = merge_boxes(detections);
    // Convert detections to original image size

    for (auto &box : merged_boxes)
    {
        box.x1 = static_cast<int>(box.x1 * width / 320);
        box.y1 = static_cast<int>(box.y1 * height / 320);
        box.x2 = static_cast<int>(box.x2 * width / 320);
        box.y2 = static_cast<int>(box.y2 * height / 320);
    }
    return merged_boxes;
}

void process_fusion(const ros::TimerEvent &event)
{
    cv::Mat rgb_copy;
    {
        std::lock_guard<std::mutex> lock(buffer_mutex);
        if (rgb_image.empty())
        {
            ROS_WARN_THROTTLE(1, "Waiting for RGB images.");
            return;
        }
        rgb_copy = rgb_image.clone();
    }

    // Detect objects in the RGB image
    std::vector<BoxInfo> detections = detection_pipeline(rgb_copy);
    if (detections.empty())
    {
        ROS_WARN_THROTTLE(1, "No detections found.");
        return;
    }
    // Draw detections on the image
    for (const auto &box : detections)
    {
        cv::rectangle(rgb_copy, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(0, 255, 0), 2);
    }
    std_msgs::Header header;
    header.frame_id = "camera_rgb_optical_frame";
    header.stamp = time_stamp;
    image_pub.publish(cv_bridge::CvImage(header, "bgr8", rgb_copy).toImageMsg());

    // Publish reduced detections
    cavity_detection_msgs::LogoObservation msg;
    msg.header.frame_id = "camera_link";
    msg.header.stamp = time_stamp;
    for (const auto &box : detections)
    {
        msg.points.push_back(box.x1);
        msg.points.push_back(box.y1);
        msg.points.push_back(box.x2);
        msg.points.push_back(box.y2);
    }
    vert_pub.publish(msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vertical_detector_node");
    ros::NodeHandle nh;

    // Subscribers
    ros::Subscriber rgb_sub = nh.subscribe("/camera/rgb/image_raw", 1, rgb_callback);
    // Publishers
    vert_pub = nh.advertise<cavity_detection_msgs::LogoObservation>("/vert_logo", 10);
    image_pub = nh.advertise<sensor_msgs::Image>("/cavity_detection/logo_image", 10);

    // Set up periodic processing
    ros::Timer timer = nh.createTimer(ros::Duration(1.0 / RUN_RATE), process_fusion);

    ros::spin();
    return 0;
}
