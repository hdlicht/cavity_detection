cmake_minimum_required(VERSION 3.0.2)
project(cavity_detection)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
set(ncnn_DIR "../../ncnn/build/install/lib/cmake/ncnn")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  roscpp
  std_msgs
  message_generation
  cavity_detection_msgs
  ncnn
  cv_bridge
  sensor_msgs
  image_transport
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
)
find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  CATKIN_DEPENDS geometry_msgs rospy std_msgs message_runtime roscpp tf2_ros
#  INCLUDE_DIRS include
#  LIBRARIES cavity_detection
#  CATKIN_DEPENDS geometry_msgs rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########



## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ../../ncnn/src  
  ../../ncnn/build/install/include
  ../../ncnn/build/install/include/ncnn

)

## Declare a C++ library
add_library(logo_detector
  src/logo_detector/detector.cpp
)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(vertical_roi src/vertical_roi.cpp)
target_link_libraries(vertical_roi 
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  logo_detector
  ncnn
)

add_executable(vertical_servo src/vertical_servo.cpp)
target_link_libraries(vertical_servo 
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  logo_detector
  ncnn
)

# add_executable(map_maker_node src/map_maker.cpp)
# add_dependencies(map_maker_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
# target_link_libraries(map_maker_node
#   ${catkin_LIBRARIES}
#   costmap_2d
# )

## Specify libraries to link a library or executable target against
target_link_libraries(
  logo_detector
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ncnn

)
