#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 12:31:49 2024

@author: ahsanjalal
"""

import os
import pandas as pd

# Define paths
image_dir = "img_data_new"
csv_path = "lengths_combined_new.csv"

# Load the CSV file
df = pd.read_csv(csv_path)

# Filter rows based on Range column values
filtered_df = df[(df['Range'] > 1000) & (df['Range'] < 5000)]

# Get the list of image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Process each image
for image_file in image_files:
    # Check if the image name is in the OzFishFrame column
    if image_file in filtered_df['OzFishFrame'].values:
        # Get the corresponding row
        row = filtered_df[filtered_df['OzFishFrame'] == image_file]

        # If Range condition is not satisfied, remove the file
        if row.empty:
            print(f"Removing file (not in range): {image_file}")
            os.remove(os.path.join(image_dir, image_file))
            txt_file = os.path.join(image_dir, os.path.splitext(image_file)[0] + ".txt")
            if os.path.exists(txt_file):
                os.remove(txt_file)
    else:
        # If the file is not in the OzFishFrame column, remove it
        print(f"Removing file (not found in OzFishFrame): {image_file}")
        os.remove(os.path.join(image_dir, image_file))
        txt_file = os.path.join(image_dir, os.path.splitext(image_file)[0] + ".txt")
        if os.path.exists(txt_file):
            os.remove(txt_file)
