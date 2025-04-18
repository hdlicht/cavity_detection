import subprocess
import time

def stop_ros_node(node_name):
    """Stops a running ROS node by name."""
    try:
        subprocess.run(["rosnode", "kill", node_name], check=True)
        print(f"Node {node_name} killed successfully.")
    except subprocess.CalledProcessError:
        print(f"Failed to kill node {node_name}. It might not be running.")

def start_ros_node(package, executable):
    """Starts a new ROS node."""
    try:
        subprocess.Popen(["rosrun", package, executable])
        print(f"Started {executable} from {package}.")
    except Exception as e:
        print(f"Failed to start node: {e}")

if __name__ == "__main__":
    cavity_node = "/cavity_detector"  # Replace with your actual node name
    new_package = "cavity_processing"  # Replace with the package name
    new_executable = "process_cavities.py"  # Replace with the actual node script

    stop_ros_node(cavity_node)
    time.sleep(2)  # Wait for shutdown to complete
    start_ros_node(new_package, new_executable)
