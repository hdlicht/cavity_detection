#!/usr/bin/env python

import rosbag
import csv
import rospy
import sys

def bag_to_csv(bag_file, csv_file):
    with rosbag.Bag(bag_file, 'r') as bag, open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "topic", "message_type"])  # CSV header

        for topic, msg, t in bag.read_messages():
            msg_type = msg._type if hasattr(msg, "_type") else "Unknown"
            writer.writerow([t.to_sec(), topic, msg_type])

    print(f"Exported {bag_file} to {csv_file}")

if __name__ == "__main__":

    bag_to_csv('bags/moving1.bag', 'bags/moving1.csv')
