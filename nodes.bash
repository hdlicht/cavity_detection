#!/bin/bash

# Start tmux session
tmux new-session -d -s cavity_detection  # Create a new tmux session in detached mode

# ROI detector nodes
tmux send-keys -t cavity_detection:0 "robot_ros && roslaunch cavity_detection roi.launch" C-m

# Map Maker Node (vertical location and time-in-sight map)
tmux split-window -h
tmux send-keys -t cavity_detection:0.1 "robot_ros && rosrun cavity_detection map_maker.py" C-m

# Cavity Server
tmux split-window -h
tmux send-keys -t cavity_detection:0.2 "robot_ros && rosrun cavity_detection cavity_map_server_2.py" C-m

# Vertical Servoing Node
tmux split-window -h
tmux send-keys -t cavity_detection:0.2 "robot_ros && rosrun cavity_detection vertical_servo.py" C-m

# Attach to the tmux session to view the result
tmux attach -t cavity_detection