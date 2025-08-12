#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 15:45:07 2025

@author: ahsanjalal
"""

import os
import cv2
import csv
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from depth_anything_v2.dpt import DepthAnythingV2
from itertools import islice

# === CONFIG ===
IMG_DIR = '../data/test_images_ozfish'
GT_CSV = '../data/test_ozfish.csv'
YOLO_MODEL = '../data/yolov11_ozfish.pt'
DEPTH_MODEL = 'checkpoints/depth_anything_v2_vitb.pth'
SAVE_CSV = '../data/estimated_lengths_ozfish.csv'


INPUT_SIZE = 518
ENCODER = 'vitb'

# === INIT MODELS ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model = YOLO(YOLO_MODEL)

model_configs = {
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
}
depth_model = DepthAnythingV2(**model_configs[ENCODER])
depth_model.load_state_dict(torch.load(DEPTH_MODEL, map_location=device))
depth_model.to(device).eval()

# === LOAD GT CSV ===
df = pd.read_csv(GT_CSV)
df["ImageKey"] = df["FilenameLeft"] + "." + df["FrameLeft"].astype(str)
grouped = df.groupby(["ImageKey", "ImagePtPair"])  # Group by image and point pair
results = []
missed_yolo = 0

# === PROCESS EACH IMAGE ===
for (image_key, point_pair), group in tqdm(grouped, total=len(grouped), desc="Processing"):
    img_file = os.path.join(IMG_DIR, f"{image_key}.png")
    txt_file = os.path.join(IMG_DIR, f"{image_key}.txt")
    
    if not os.path.exists(img_file):
        continue
        
    image = cv2.imread(img_file)
    if image is None:
        continue
    h, w = image.shape[:2]

    # Get depth map
    with torch.no_grad():
        depth_raw = depth_model.infer_image(image, INPUT_SIZE)

    # Convert depth to mm (adjust these values based on your depth model)
    max_depth = np.max(depth_raw)
    zmax, zmin = 3500, 800
    depth_mm_map = zmax - depth_raw * ((zmax - zmin) / max_depth)


    # Get coordinates from CSV (two points per ImagePtPair)
    if len(group) != 2:
        continue  # Need exactly two points for each pair
        
    pt1 = group.iloc[0]
    pt2 = group.iloc[1]
    
    # Get GT coordinates (Lx, Ly columns)
    xg1, yg1 = int(pt1['Lx']), int(pt1['Ly'])
    xg2, yg2 = int(pt2['Lx']), int(pt2['Ly'])
    
    # Calculate midpoint and length
    center_x = (xg1 + xg2) // 2
    center_y = (yg1 + yg2) // 2
    gt_px_len=np.sqrt((xg2 - xg1)**2 + (yg2 - yg1)**2)
    # gt_px_len = math.hypot(xg2 - xg1, yg2 - yg1)
    mid_z = abs(pt1['MidZ'])  # Take absolute value of depth
    
    # Get YOLO predictions (either from model or txt file)
    yolo_boxes = []
  
    yolo_results = yolo_model(img_file)[0]
    if yolo_results.boxes is None:
        missed_yolo += 1
        print(f'Total missed files by YOLO: {missed_yolo}/{len(grouped)}')
        continue
        
    boxes = yolo_results.boxes.xywhn.cpu().numpy()
    for i, (cls, xc, yc, bw, bh) in enumerate(zip(yolo_results.boxes.cls.cpu().numpy(), *boxes.T)):
        xc_abs, yc_abs = int(xc * w), int(yc * h)
        bw_abs, bh_abs = int(bw * w), int(bh * h)
        x1, y1 = int(xc_abs - bw_abs / 2), int(yc_abs - bh_abs / 2)
        x2, y2 = int(xc_abs + bw_abs / 2), int(yc_abs + bh_abs / 2)
        yolo_boxes.append((x1, y1, x2, y2))

    # Process each YOLO box
    for box in yolo_boxes:
        x1, y1, x2, y2 = box
        xc_abs = (x1 + x2) // 2
        yc_abs = (y1 + y2) // 2
        bw_abs = x2 - x1
        bh_abs = y2 - y1
        
        if 0 <= xc_abs < w and 0 <= yc_abs < h:
            # Estimate depth using ellipse or Circle of radius =10% of box width
            box_size = max(bw_abs, bh_abs)
            long_axis = int(0.05 * box_size)
            short_axis = int(0.1 * box_size)
            
            ellipse_mask = np.zeros(depth_mm_map.shape, dtype=np.uint8)
            cv2.ellipse(ellipse_mask, (xc_abs, yc_abs), (long_axis, short_axis), 0, 0, 360, 255, -1)
            region_values = depth_mm_map[ellipse_mask == 255]
            box_depth_old = float(np.mean(region_values))
            box_depth = box_depth_old + 100  #  adjustment

            # Estimate lengthGT_CSV
            box_pixel_len=np.sqrt((x2 - x1)**2 + (y2 - y1)**2)*0.9
            # box_pixel_len = max(bw_abs, bh_abs)* 0.9
            box_est_len_mm = box_pixel_len * (box_depth / 2000)

            # Create rotated rectangle for GT line
            angle_rad = math.atan2(yg2 - yg1, xg2 - xg1)
            angle_deg = math.degrees(angle_rad)
            long_axis_rect = gt_px_len
            short_axis_rect = 0.3 * long_axis_rect
            
            rect = ((center_x, center_y), (long_axis_rect, short_axis_rect), angle_deg)
            box_pts = cv2.boxPoints(rect)
            box_pts = np.int0(box_pts)
            gt_x1, gt_y1 = np.min(box_pts[:, 0]), np.min(box_pts[:, 1])
            gt_x2, gt_y2 = np.max(box_pts[:, 0]), np.max(box_pts[:, 1])
            
            # Calculate IOU
            inter_x1, inter_y1 = max(x1, gt_x1), max(y1, gt_y1)
            inter_x2, inter_y2 = min(x2, gt_x2), min(y2, gt_y2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
            pred_area = (x2 - x1) * (y2 - y1)
            union = gt_area + pred_area - inter_area
            iou = inter_area / union if union > 0 else 0

            if iou >= 0.5:
                # Save results
                result_row = [
                    image_key,
                    point_pair,
                    int(pt1['FrameLeft']),
                    mid_z,
                    box_depth,
                    gt_px_len,
                    box_pixel_len,
                    pt1['Length'],  # Assuming Length column exists
                    box_est_len_mm,
                    abs(pt1['Length'] - box_est_len_mm)
                ]
                
                results.append(result_row)
                print(f"""
                GT Depth (mm): {mid_z:.2f} :: Estimated Depth (mm): {box_depth:.2f}
                GT Pixel Length: {gt_px_len:.2f} ::: Estimated Pixel Length: {box_pixel_len:.2f}
                GT Fish Length (mm): {pt1['Length']:.2f} ::: Estimated Fish Length (mm): {box_est_len_mm:.2f}
                """)
              
# === SAVE RESULTS CSV ===

columns = [
    'ImageKey', 'PointPair', 'Frame', 'GT_Depth_mm', 'Est_Depth_mm',
    'GT_px_L', 'Est_px_L', 'Length', 'est_length', 'Abs_Diff_mm'
]
results_df = pd.DataFrame(results, columns=columns)
results_df.to_csv(SAVE_CSV, index=False)

print(f"Saved full analysis to: {SAVE_CSV}")
