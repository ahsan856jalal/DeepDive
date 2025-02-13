#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 13:51:28 2024

@author: ahsanjalal
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:03:50 2023

@author: ahsanjalal
"""

import os
from os.path import join
import cv2
import random
from ultralytics import YOLO

# Set deterministic random seed for bbox colors
random.seed(3)

# Directories
image_dir = "img_data_new"
save_dir = "yolov11_data_train11"

# Ensure save directory exists
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Load YOLO model
model = YOLO('/home/ahsanjalal/yolov11/runs/detect/train11/weights/best.pt')

# Get list of all PNG images
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Process images
for image_file in image_files:
    # Full path to the image
    image_path = join(image_dir, image_file)
    # Extract the image name without the extension
    img_name = image_file.split('.png')[0]

    # YOLO inference
    results = model(image_path, save=True)

    # Save YOLO results in text format
    result_txt_path = join(save_dir, f"{img_name}.txt")
    with open(result_txt_path, 'w') as file:
        for idx, prediction in enumerate(results[0].boxes.xywhn):  # YOLO normalized bbox format
            cls = int(results[0].boxes.cls[idx].item())  # Class label
            # Write line to file in YOLO label format: cls x_center y_center width height
            file.write(f"{cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}\n")

    print(f"Processed and saved results for: {image_file}")

print("YOLO inference completed.")
