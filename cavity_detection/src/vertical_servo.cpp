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

// Global shared buffer
cv::Mat rgb_image;
ros::Time time_stamp;
std::mutex buffer_mutex;
ros::Publisher image_pub;
tf::TransformListener *tf_listener;
NanoDet detector("/home/pi/capstone-vision/src/cavity_detection/cavity_detection/model/nanodet.param", 
    "/home/pi/capstone-vision/src/cavity_detection/cavity_detection/model/nanodet.bin", false);

// Constants
const int RUN_RATE = 5;

                                             
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

std::vector<BoxInfo> detection_pipeline(const cv::Mat &image)
{
    int width = image.cols;
    int height = image.rows;
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(320, 320));
    std::vector<BoxInfo> detections = detector.detect(resized_image, 0.3, 0.3);
    if (detections.size() < 3)
        return {};
    // Convert detections to original image size

    for (auto &box : detections)
    {
        box.x1 = static_cast<int>(box.x1 * width / 320);
        box.y1 = static_cast<int>(box.y1 * height / 320);
        box.x2 = static_cast<int>(box.x2 * width / 320);
        box.y2 = static_cast<int>(box.y2 * height / 320);
    }
    return detections;
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

    // Perform detection
    std::vector<BoxInfo> detections = detection_pipeline(rgb_copy);

    if (!detections.empty())
    {
        cv::Mat mask = cv::Mat::zeros(rgb_copy.size(), CV_8UC1);
        // Do inpainting on the detected boxes to remove them
        for (const auto &box : detections)
        {
            cv::rectangle(mask, cv::Point(box.x1, box.y1), cv::Point(box.x2, box.y2), cv::Scalar(255), -1);
        }
        cv::inpaint(rgb_copy, mask, rgb_copy, 3, cv::INPAINT_TELEA);
    }
    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(rgb_copy, gray, cv::COLOR_BGR2GRAY);

    // Histogram Equalization (optional contrast enhancement)
    cv::equalizeHist(gray, gray);

    // Sobel Edge Detection in x-direction (vertical edges)
    cv::Mat grad_x;
    cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3);
    cv::convertScaleAbs(grad_x, grad_x);

    // Threshold to isolate strong edges
    cv::Mat binary;
    cv::threshold(grad_x, binary, 30, 255, cv::THRESH_BINARY);

    // Morphological operations to clean and connect lines
    cv::Mat morph;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 15));
    cv::morphologyEx(binary, morph, cv::MORPH_CLOSE, kernel);

    // Convert to ROS message
    sensor_msgs::ImagePtr img_msg = cv_bridge::CvImage(std_msgs::Header(), "mono8", morph).toImageMsg();
    img_msg->header.stamp = time_stamp;
    img_msg->header.frame_id = "camera_rgb_frame";
    image_pub.publish(img_msg);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vertical_detector_node");
    ros::NodeHandle nh;

    // Subscribers
    ros::Subscriber rgb_sub = nh.subscribe("/camera/rgb/image_raw", 1, rgb_callback);
    // Publishers
    image_pub = nh.advertise<sensor_msgs::Image>("/cavity_detection/logo_image", 10);

    // Set up periodic processing
    ros::Timer timer = nh.createTimer(ros::Duration(1.0 / RUN_RATE), process_fusion);

    ros::spin();
    return 0;
}
