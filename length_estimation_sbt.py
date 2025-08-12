#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 12:07:01 2025

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


os.chdir('Depth-Anything-V2')
#move to depth anytyhing v2 folder, it should be in DeepDive main folder

IMG_DIR = '../data/test_images_sbt'
GT_CSV = '../data/test_sbt.csv'
YOLO_MODEL='../data/yolov11_sbt.pt'
DEPTH_MODEL = 'checkpoints/depth_anything_v2_vitb.pth'
SAVE_CSV = '../data/estimated_lengths_sbt.csv'

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
df["ImageKey"] = df["Filename"] + "_" + df["Frame"].astype(str)
grouped = df.groupby("ImageKey")
# grouped = dict(islice(grouped, 100)) 
results = []
missed_yolo=0
# === PROCESS EACH IMAGE ===
for image_key, group in tqdm(grouped, total=len(grouped), desc="Processing"):
    img_file = os.path.join(IMG_DIR, f"{image_key}.png")
    print(image_key)
    if not os.path.exists(img_file):
        continue
    print(image_key)
    image = cv2.imread(img_file)
    if image is None:
        continue
    h, w = image.shape[:2]

    with torch.no_grad():
        depth_raw = depth_model.infer_image(image, INPUT_SIZE)


    max_depth = np.max(depth_raw)
    zmax, zmin = 3500, 800
    depth_mm_map = zmax - depth_raw * ((zmax - zmin) / max_depth)

   
    # YOLO Inference
    yolo_results = yolo_model(img_file)[0]
    if yolo_results.boxes is None:
        missed_yolo+=1
        print(f'total missed files by yolo are {missed_yolo}/{len(grouped)}')
        continue

    boxes = yolo_results.boxes.xywhn.cpu().numpy()
    for i, (cls, xc, yc, bw, bh) in enumerate(zip(yolo_results.boxes.cls.cpu().numpy(), *boxes.T)):
        xc_abs, yc_abs = int(xc * w), int(yc * h)
        bw_abs, bh_abs = int(bw * w), int(bh * h)
        x1, y1 = int(xc_abs - bw_abs / 2), int(yc_abs - bh_abs / 2)
        x2, y2 = int(xc_abs + bw_abs / 2), int(yc_abs + bh_abs / 2)

        if 0 <= xc_abs < w and 0 <= yc_abs < h:
            # Estimate depth using ellipse or can use 10% of box width as circle radius
            box_size = max(bw_abs, bh_abs)
            long_axis = int(0.05 * box_size)
            short_axis = int(0.1 * box_size)
            ellipse_mask = np.zeros(depth_mm_map.shape, dtype=np.uint8)
            cv2.ellipse(ellipse_mask, (xc_abs, yc_abs), (long_axis, short_axis), 0, 0, 360, 255, -1)
            region_values = depth_mm_map[ellipse_mask == 255]
            box_depth_old = float(np.mean(region_values))
            box_depth=box_depth_old


            # Estimate length
            
            box_pixel_len = max(bw_abs, bh_abs) *0.9# 
            box_est_len_mm = box_pixel_len * (box_depth / 1120)

            # === GET GT LINE ===
            row = group.iloc[0]
            xg1, yg1 = int(row['Left pt1 col (px)']), int(row['Left pt1 row (px)'])
            xg2, yg2 = int(row['Left pt2 col (px)']), int(row['Left pt2 row (px)'])
            gt_px_len = math.hypot(xg2 - xg1, yg2 - yg1)
            mid_z = abs(row['Mid Z (mm)'])
            gt_mm = row['Length (mm)']
                
            center_x = (xg1 + xg2) // 2
            center_y = (yg1 + yg2) // 2
            angle_rad = math.atan2(yg2 - yg1, xg2 - xg1)
            angle_deg = math.degrees(angle_rad)
            long_axis = gt_px_len
            short_axis = 0.3 * long_axis
            
            # Construct rotated rectangle
            rect = ((center_x, center_y), (long_axis, short_axis), angle_deg)
            box_pts = cv2.boxPoints(rect)
            box_pts = np.int0(box_pts)
            gt_x1, gt_y1 = np.min(box_pts[:, 0]), np.min(box_pts[:, 1])
            gt_x2, gt_y2 = np.max(box_pts[:, 0]), np.max(box_pts[:, 1])
            
         
            # === IOU/Overlap Check ===
            inter_x1, inter_y1 = max(x1, gt_x1), max(y1, gt_y1)
            inter_x2, inter_y2 = min(x2, gt_x2), min(y2, gt_y2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
            pred_area = (x2 - x1) * (y2 - y1)
            union = gt_area + pred_area - inter_area
            iou = inter_area / union if union > 0 else 0

            if iou >= 0.5:
               
                # === Save depth + ellipse ===
                result_row = [
                image_key,
                int(row['Frame']),
                mid_z,
                box_depth,
                gt_px_len,
                box_pixel_len,
                gt_mm,
                box_est_len_mm,
                abs(gt_mm - box_est_len_mm)]
              

                results.append(result_row)
                # === Append results ===
                # results.append([
                #     image_key,
                #     int(row['Frame']),
                #     mid_z,
                #     box_depth,
                #     gt_px_len,
                #     box_pixel_len,
                #     gt_mm,
                #     box_est_len_mm,
                #     abs(gt_mm - box_est_len_mm)
                # ])
            # else:
                # missed_yolo+=1
                # print(f'yolo prediction donot overlaps: {missed_yolo}')

# === SAVE RESULTS CSV ===
columns = [
    'ImageKey', 'Frame', 'GT_Depth_mm', 'Est_Depth_mm',
    'GT_px_L', 'Est_px_L', 'Length', 'est_length', 'Abs_Diff_mm'
]
with open(SAVE_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerows(results)

print(f"Saved full analysis to: {SAVE_CSV}")
