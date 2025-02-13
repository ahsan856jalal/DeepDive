#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:20:42 2024

@author: ahsanjalal
"""

import os
import cv2
import math

# Paths
base_dir = "img_data_new"
image_dir = base_dir
annot_dir = base_dir

# Function to parse YOLO format annotations
def parse_yolo_annotations(txt_file, img_width, img_height):
    boxes = []
    with open(txt_file, 'r') as f:
        for line in f:
            # YOLO format: class x_center y_center width height
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x_center, y_center, width, height = map(float, parts)
            # Convert YOLO normalized values to pixel coordinates
            x_min = int((x_center - width / 2) * img_width)
            x_max = int((x_center + width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            y_max = int((y_center + height / 2) * img_height)
            boxes.append((x_min, y_min, x_max, y_max))
    return boxes

# Function to calculate Euclidean distance
def calculate_euclidean_distance(x_min, y_min, x_max, y_max):
    return math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)

# Loop through all images in the directory
for file in os.listdir(image_dir):
    if file.endswith('.png'):
        # Read image
        img_path = os.path.join(image_dir, file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Failed to load image: {file}")
            continue
        
        img_height, img_width = image.shape[:2]

        # Read corresponding annotation file
        txt_file = os.path.join(annot_dir, file.replace('.png', '.txt'))
        if not os.path.exists(txt_file):
            print(f"No annotation file found for {file}")
            continue

        # Parse YOLO annotations
        boxes = parse_yolo_annotations(txt_file, img_width, img_height)
        for (x_min, y_min, x_max, y_max) in boxes:
            distance = calculate_euclidean_distance(x_min, y_min, x_max, y_max)
            if distance < 50:
                print(f"Removing {file} and its annotation (distance = {distance:.2f})")
                # Remove the image and text file
                # os.remove(img_path)
                # os.remove(txt_file)
                break  # Move to the next image once a distance < 110 is found
