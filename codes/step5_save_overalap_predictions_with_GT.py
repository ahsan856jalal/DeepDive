#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:07:14 2024

@author: ahsanjalal
"""

import os
from os.path import join
import numpy as np

# Directories
gt_dir = "img_data_new"
pred_dir = "yolov11_data_train11"
output_dir = "yolov11_overlap_train11"

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to calculate IOU
def calculate_iou(box1, box2):
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Iterate through ground truth files
gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]

for gt_file in gt_files:
    gt_path = join(gt_dir, gt_file)
    pred_path = join(pred_dir, gt_file)

    # Skip if corresponding prediction file doesn't exist
    if not os.path.exists(pred_path):
        print(f"Prediction file not found for: {gt_file}")
        continue

    # Read GT boxes
    with open(gt_path, 'r') as gt_f:
        gt_boxes = [list(map(float, line.split()[1:])) for line in gt_f.readlines()]

    # Read prediction boxes
    with open(pred_path, 'r') as pred_f:
        pred_boxes = [list(map(float, line.split()[1:])) for line in pred_f.readlines()]

    filtered_predictions = []
    for gt_box in gt_boxes:
        max_iou = 0
        best_pred = None
        for pred_box in pred_boxes:
            iou = calculate_iou(gt_box, pred_box)
            if iou > max_iou:
                max_iou = iou
                best_pred = pred_box
        if best_pred:
            # Keep the best prediction with class label
            cls_label = pred_boxes[pred_boxes.index(best_pred)][0]
            filtered_predictions.append([cls_label] + best_pred)

    # Save filtered predictions
    output_path = join(output_dir, gt_file)
    with open(output_path, 'w') as out_f:
        for pred in filtered_predictions:
            out_f.write(" ".join(map(str, pred)) + "\n")

    print(f"Processed: {gt_file}")

print("Filtering complete.")
