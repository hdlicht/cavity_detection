#include <ros/ros.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Point.h>
#include <costmap_2d/costmap_2d.h>
#include <costmap_2d/cost_values.h>
#include <cavity_detection_msgs/VerticalObservation.h>

#include <vector>
#include <cmath>
#include <set>
#include <algorithm> // For std::min, std::max

const int FREE_CELL = 0;
const int OCCUPIED = 100;
const int UNKNOWN = -1;


// Structure to hold camera intrinsics (can be loaded from params)
struct CameraIntrinsics {
    double fx = 570.342;
    double fy = 570.342;
    double cx = 314.5;
    double cy = 235.5;
};

// Structure for 2D points (simple)
struct Point2D {
    double x;
    double y;
};

// Helper function for point in triangle test (using cross products)
// Checks if point P is on the same side of edges AB, BC, CA
double sign(Point2D p1, Point2D p2, Point2D p3) {
    return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
}

bool isPointInTriangle(Point2D pt, Point2D v1, Point2D v2, Point2D v3) {
    double d1, d2, d3;
    bool has_neg, has_pos;

    d1 = sign(pt, v1, v2);
    d2 = sign(pt, v2, v3);
    d3 = sign(pt, v3, v1);

    has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

    // Point is inside if all signs are the same (or zero)
    return !(has_neg && has_pos);
}

class MapMaker {
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;

    // ROS Communication
    ros::Subscriber map_sub_;
    ros::Subscriber roi_sub_;
    ros::Publisher grid_pub_;
    ros::Publisher marker_pub_; // Optional visualization

    // TF
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // Costmap for Obstacle Checks
    costmap_2d::Costmap2D obstacle_map_; // Holds /map data
    bool obstacle_map_received_ = false;

    // Logo Observation Grid (Output)
    nav_msgs::OccupancyGrid grid_;
    bool grid_initialized_ = false;

    // Parameters
    CameraIntrinsics k_rgb_;
    int initial_logo_value_;
    int increment_;
    int decrement_; // Currently unused, but for future extension
    int obstacle_threshold_;
    double fixed_depth_;
    std::string map_frame_ = "map"; // Make configurable

public:
    MapMaker() :
        private_nh_("~"),
        tf_listener_(tf_buffer_)
    {
        // Load Parameters
        private_nh_.param<int>("initial_logo_value", initial_logo_value_, 50);
        private_nh_.param<int>("increment", increment_, 10);
        private_nh_.param<int>("decrement", decrement_, 5); // For later
        private_nh_.param<int>("obstacle_threshold", obstacle_threshold_, 90); // Threshold on /map cost
        private_nh_.param<double>("fixed_depth", fixed_depth_, 5.0);
        private_nh_.param<std::string>("map_frame", map_frame_, "map");
        // Could load K_rgb_ values from params too

        // Initialize Subscribers & Publishers
        map_sub_ = nh_.subscribe("/map", 1, &MapMaker::mapCallback, this);
        roi_sub_ = nh_.subscribe("/vert_roi", 10, &MapMaker::roiCallback, this); // Increased queue size
        grid_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("/observation_grid", 1);
        marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/markers", 1); // Optional
    }

    // Callback for receiving the OccupancyGrid used for obstacle checks
    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg) {
        // Update the internal obstacle Costmap2D object
        if (!obstacle_map_received_ ||
            obstacle_map_.getSizeInCellsX() != msg->info.width ||
            obstacle_map_.getSizeInCellsY() != msg->info.height ||
            std::abs(obstacle_map_.getResolution() - msg->info.resolution) > 1e-6 ||
            obstacle_map_.getOriginX() != msg->info.origin.position.x ||
            obstacle_map_.getOriginY() != msg->info.origin.position.y)
        {
            obstacle_map_.resizeMap(msg->info.width, msg->info.height, msg->info.resolution,
                                    msg->info.origin.position.x, msg->info.origin.position.y);
            ROS_INFO("Obstacle map resized/initialized.");

            // Also initialize/resize the logo grid to match
            grid_.info = msg->info; // Copy metadata
            grid_.data.assign(msg->info.width * msg->info.height, initial_logo_value_);
            grid_initialized_ = true;
            ROS_INFO("observation grid initialized/resized.");
        }

        // Copy data, converting from OccupancyGrid (-1, 0-100) to Costmap2D (0-255)
        unsigned char* char_map = obstacle_map_.getCharMap();
        size_t map_size = obstacle_map_.getSizeInCellsX() * obstacle_map_.getSizeInCellsY();

        if (msg->data.size() == map_size) {
            for (size_t i = 0; i < map_size; ++i) {
                if (msg->data[i] == -1) {
                    char_map[i] = UNKNOWN;
                } else if (msg->data[i] >= obstacle_threshold_) { // Treat >= threshold as LETHAL
                    char_map[i] = OCCUPIED;
                } else if (msg->data[i] > 0) { // Treat other costs as obstacles too (simple)
                    // Could map differently, e.g., to INSCRIBED
                     char_map[i] = OCCUPIED; // Simpler: any cost blocks ray
                } else { // 0
                    char_map[i] = FREE_CELL;
                }
            }
        } else {
            ROS_ERROR("Received map data size (%zu) does not match expected size (%zu)",
                      msg->data.size(), map_size);
            return; // Don't set received flag if data is bad
        }
        obstacle_map_received_ = true;
    }

    // Check if line of sight is blocked using internal obstacle map
    bool isOccluded(double start_map_x, double start_map_y, double end_map_x, double end_map_y) {
        if (!obstacle_map_received_) {
            return true; // No map, assume occluded
        }

        unsigned int start_x_uint = static_cast<unsigned int>(start_map_x);
        unsigned int start_y_uint = static_cast<unsigned int>(start_map_y);
        unsigned int end_x_uint = static_cast<unsigned int>(end_map_x);
        unsigned int end_y_uint = static_cast<unsigned int>(end_map_y);

        std::vector<costmap_2d::MapLocation> cell_ray;
        obstacle_map_.raytraceLine(cell_ray, start_x_uint, start_y_uint, end_x_uint, end_y_uint);

        // Check cells along the ray (excluding start, maybe end depending on exact need)
        for (size_t i = 0; i < cell_ray.size(); ++i) { // Check all cells returned by raytrace
             // Avoid checking the exact start/end cells if desired? Usually check intermediates.
             if(cell_ray[i].x == start_x_uint && cell_ray[i].y == start_y_uint) continue;
             if(cell_ray[i].x == end_x_uint && cell_ray[i].y == end_y_uint) continue; // Don't check destination cell itself

             unsigned char cost = obstacle_map_.getCost(cell_ray[i].x, cell_ray[i].y);
             // If cost is LETHAL (or maybe INSCRIBED), it's occluded
             if (cost >= OCCUPIED) { 
                 return true;
             }
             // Optional: Treat NO_INFORMATION as occluding? Depends on desired behavior.
             // if (cost == costmap_2d::NO_INFORMATION) return true;
        }
        return false; // No lethal obstacles found along the ray
    }


    // Callback for VerticalObservation ROI messages
    void roiCallback(const cavity_detection_msgs::VerticalObservation::ConstPtr& msg) {
        if (!obstacle_map_received_ || !grid_initialized_) {
            ROS_WARN_THROTTLE(5.0, "Map data or logo grid not yet initialized.");
            return;
        }

        geometry_msgs::TransformStamped tf_cam_to_map;
        try {
            tf_cam_to_map = tf_buffer_.lookupTransform(map_frame_, msg->header.frame_id, msg->header.stamp, ros::Duration(0.2)); // Shorter timeout
        } catch (tf2::TransformException &ex) {
            ROS_WARN("TF lookup failed from %s to %s: %s", map_frame_.c_str(), msg->header.frame_id.c_str(), ex.what());
            return;
        }

        // Camera position in map frame (origin of FoV cone)
        double cam_x_world = tf_cam_to_map.transform.translation.x;
        double cam_y_world = tf_cam_to_map.transform.translation.y;
        unsigned int cam_x_map, cam_y_map; // Map coordinates (unsigned int)
        if (!obstacle_map_.worldToMap(cam_x_world, cam_y_world, cam_x_map, cam_y_map)) {
            ROS_WARN("Camera center is outside the map bounds.");
            return;
        }

        std::set<std::pair<unsigned int, unsigned int>> visible_cells_in_fov;
        visualization_msgs::MarkerArray marker_array; // For optional visualization

        // --- Process each ROI segment ---
        const std::vector<int>& points = msg->points;
        // Expect points to be [left1, top1, right1, bottom1, left2, top2, right2, bottom2, ...]
        if (points.size() % 4 != 0) {
             ROS_ERROR("Invalid points vector size in VerticalObservation message.");
             return;
        }

        for (size_t i = 0; i < points.size(); i += 4) {
            int left_pixel = points[i];
            int right_pixel = points[i + 2];
            // top/bottom pixels ignored in this 2D projection

            // Calculate FoV triangle vertices in camera optical frame (X-right, Y-down, Z-forward)
            // Project pixels to 3D points at fixed_depth_
            double X_left = (left_pixel - k_rgb_.cx) / k_rgb_.fx * fixed_depth_;
            double X_right = (right_pixel - k_rgb_.cx) / k_rgb_.fx * fixed_depth_;
            // Y coordinates are zero as we project onto the camera's XZ plane for the 2D map FoV
            geometry_msgs::PointStamped v0_cam, v1_cam, v2_cam;
            v0_cam.header.frame_id = msg->header.frame_id;
            v0_cam.header.stamp = msg->header.stamp; // Use stamp from message
            v0_cam.point.x = 0.0; v0_cam.point.y = 0.0; v0_cam.point.z = 0.0; // Origin

            v1_cam.header = v0_cam.header;
            v1_cam.point.x = X_left; v1_cam.point.y = 0.0; v1_cam.point.z = fixed_depth_; // Left point

            v2_cam.header = v0_cam.header;
            v2_cam.point.x = X_right; v2_cam.point.y = 0.0; v2_cam.point.z = fixed_depth_; // Right point

            // Transform vertices to map frame
            geometry_msgs::PointStamped v0_map, v1_map, v2_map;
            try {
                tf_buffer_.transform(v0_cam, v0_map, map_frame_);
                tf_buffer_.transform(v1_cam, v1_map, map_frame_);
                tf_buffer_.transform(v2_cam, v2_map, map_frame_);
            } catch (tf2::TransformException &ex) {
                ROS_WARN("TF transform failed for FoV vertices: %s", ex.what());
                continue; // Skip this ROI segment
            }

            // Store vertices for polygon check and markers
            Point2D tri_v0 = {v0_map.point.x, v0_map.point.y};
            Point2D tri_v1 = {v1_map.point.x, v1_map.point.y};
            Point2D tri_v2 = {v2_map.point.x, v2_map.point.y};

            // --- Find visible cells within this triangle ---
            // Calculate bounding box in map coordinates
            double min_wx = std::min({tri_v0.x, tri_v1.x, tri_v2.x});
            double max_wx = std::max({tri_v0.x, tri_v1.x, tri_v2.x});
            double min_wy = std::min({tri_v0.y, tri_v1.y, tri_v2.y});
            double max_wy = std::max({tri_v0.y, tri_v1.y, tri_v2.y});

            unsigned int min_mx, min_my, max_mx, max_my;
            if (!obstacle_map_.worldToMap(min_wx, min_wy, min_mx, min_my) ||
                !obstacle_map_.worldToMap(max_wx, max_wy, max_mx, max_my)) {
                ROS_WARN("FoV Triangle extends outside map bounds during bounding box calculation.");
                continue;
            }
            // Ensure bounds are valid after conversion
            max_mx = std::min(max_mx, obstacle_map_.getSizeInCellsX() - 1);
            max_my = std::min(max_my, obstacle_map_.getSizeInCellsY() - 1);
            min_mx = std::max(min_mx, 0u); // Use 0u for unsigned int 0 literal
            min_my = std::max(min_my, 0u);


            for (unsigned int mx = min_mx; mx <= max_mx; ++mx) {
                for (unsigned int my = min_my; my <= max_my; ++my) {
                    // Convert cell center to world coordinates for geometric check
                    double wx, wy;
                    obstacle_map_.mapToWorld(mx, my, wx, wy); // Gets center by default
                    Point2D cell_center = {wx, wy};

                    // Check if cell center is geometrically inside the FoV triangle
                    if (isPointInTriangle(cell_center, tri_v0, tri_v1, tri_v2)) {
                        // Check if the cell is occluded by an obstacle
                         if (!isOccluded(cam_x_map, cam_y_map, mx, my)) {
                             visible_cells_in_fov.insert({mx, my});
                         }
                    }
                }
            }

            // --- Optional: Add Triangle Marker ---
            visualization_msgs::Marker triangle_marker;
            triangle_marker.header.frame_id = map_frame_;
            triangle_marker.header.stamp = ros::Time::now();
            triangle_marker.ns = "fov_triangle";
            triangle_marker.id = i;
            triangle_marker.type = visualization_msgs::Marker::LINE_STRIP;
            triangle_marker.action = visualization_msgs::Marker::ADD;
            triangle_marker.pose.orientation.w = 1.0;
            triangle_marker.scale.x = 0.02; // Line width
            triangle_marker.color.a = 0.8;
            triangle_marker.color.r = 0.0;
            triangle_marker.color.g = 1.0;
            triangle_marker.color.b = 0.0;
            triangle_marker.points.push_back(v0_map.point);
            triangle_marker.points.push_back(v1_map.point);
            triangle_marker.points.push_back(v2_map.point);
            triangle_marker.points.push_back(v0_map.point); // Close the triangle
            marker_array.markers.push_back(triangle_marker);

        } // End loop over ROI segments

        // --- Update Logo Observation Grid ---
        // Need to access logo_grid_.data as non-const
        std::vector<int8_t>& grid_data = grid_.data;
        size_t grid_width = grid_.info.width;

        for (const auto& cell_coords : visible_cells_in_fov) {
            unsigned int mx = cell_coords.first;
            unsigned int my = cell_coords.second;
            unsigned int index = obstacle_map_.getIndex(mx, my); // Use Costmap2D's index function

            if (index < grid_data.size()) { // Bounds check
                 int current_val = static_cast<int>(grid_data[index]); // Cast from int8_t
                 // Increment logic (add decrement later if needed)
                 int new_val = std::min(100, current_val + increment_); // Cap at 100 for OccupancyGrid standard range
                 grid_data[index] = static_cast<int8_t>(new_val);
            }
        }

        // --- Publish ---
        grid_.header.stamp = ros::Time::now(); // Update timestamp
        grid_.header.frame_id = map_frame_;
        grid_pub_.publish(grid_);

        if (!marker_array.markers.empty()) {
            marker_pub_.publish(marker_array);
        }
    }
}; // End class LogoMapMakerCpp


int main(int argc, char** argv) {
    ros::init(argc, argv, "map_maker_node");
    MapMaker node;
    ros::spin();
    return 0;
}