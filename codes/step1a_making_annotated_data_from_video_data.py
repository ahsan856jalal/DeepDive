#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:31:56 2024

@author: ahsanjalal
"""

import pandas as pd
import cv2
import os

# Paths
csv_path = "lengths_combined_new.csv"
image_dir = "img_data_new"
output_dir = "img_data_new"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the CSV file and extract the first 321 rows
data = pd.read_csv(csv_path)
grouped = data.groupby(["FilenameLeft", "ImagePtPair"])

for (filename_left, image_pt_pair), group in grouped:
    # Ensure only pairs are processed
    if len(group) == 2:
        row1, row2 = group.iloc[0], group.iloc[1]
        
        # Extract Lx and Ly values
        lx1, ly1 = row1["Lx"], row1["Ly"]
        lx2, ly2 = row2["Lx"], row2["Ly"]
        image_filename = f"{row1['FilenameLeft']}.{row1['FrameLeft']}.png"
        image_path = os.path.join(image_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        image_with_line = image.copy()
        cv2.line(image_with_line, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (0, 255, 0), 2)

        # Calculate bounding box coordinates
        x_min, y_min = min(lx1, lx2), min(ly1, ly2)
        x_max, y_max = max(lx1, lx2), max(ly1, ly2)
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Adjust dimensions if they are too small
        if box_height < 0.2*box_width:
            box_height = max(box_width / 2, 10)
            y_center = (y_min + y_max) / 2
            y_min = max(0, y_center - box_height / 2)
            y_max = min(height, y_center + box_height / 2)

        if box_width < 0.2*box_height:
            box_width = max(box_height / 2, 10)
            x_center = (x_min + x_max) / 2
            x_min = max(0, x_center - box_width / 2)
            x_max = min(width, x_center + box_width / 2)

        # Ensure bounding box values are within valid ranges
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(width, x_max), min(height, y_max)

        # Calculate YOLO format bbox (normalized values)
        bbox_width = (x_max - x_min) / width
        bbox_height = (y_max - y_min) / height
        x_center = (x_min + x_max) / (2 * width)
        y_center = (y_min + y_max) / (2 * height)

        # Prepare YOLO format annotation
        yolo_annotation = f"0 {x_center} {y_center} {bbox_width} {bbox_height}\n"

        # Save annotation to text file
        annotation_filename = f"{os.path.splitext(image_filename)[0]}.txt"
        annotation_path = os.path.join(output_dir, annotation_filename)
        with open(annotation_path, "w") as f:
            f.write(yolo_annotation)

        # Save the image with the line for visualization (optional)
        output_image_path = os.path.join(output_dir, f"{image_filename}")
        cv2.imwrite(output_image_path, image_with_line)

        print(f"Processed: {image_filename} -> {annotation_filename}")
