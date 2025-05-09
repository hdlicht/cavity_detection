cmake_minimum_required(VERSION 3.0.2)
project(cavity_detection_api)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  message_runtime
  roscpp
  rospy
  std_msgs
  cavity_detection_msgs
)

## Declare ROS dynamic reconfigure parameters if any (currently not needed in your case)
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

## catkin_package macro generates cmake config files for your package
catkin_package(
  CATKIN_DEPENDS message_runtime roscpp rospy std_msgs cavity_detection_msgs
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include

)

# ## C++ Code - Build the library
# add_library(${PROJECT_NAME}
#   src/cavity_detection_api/api.cpp
# )

# ## Install C++ Executables (Test Client or any other executables)
# install(TARGETS ${PROJECT_NAME}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# add_executable(test_client src/cavity_detection_api/test_client.cpp src/cavity_detection_api/api.cpp) 
# target_link_libraries(test_client
#   ${catkin_LIBRARIES}
# )

#############
## Testing ##
#############

## Add gtest-based cpp test target (if applicable for C++ code)
# catkin_add_gtest(${PROJECT_NAME}-test test/test_cavity_detection_api.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add Python tests (if any, otherwise remove or modify)
# catkin_add_nosetests(test)

catkin_python_setup()

## Install Python scripts
catkin_install_python(PROGRAMS
  scripts/cavity_detection_api/api.py
  scripts/cavity_detection_api/test_client.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)