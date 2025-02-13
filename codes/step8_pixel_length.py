#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 16:36:28 2024

@author: ahsanjalal
"""

import pandas as pd

# Load the CSV file
file_path ="combined_depth.csv"
data = pd.read_csv(file_path)

# Ensure columns fish_width and fish_height exist
if 'fish_width' not in data.columns or 'fish_height' not in data.columns:
    raise ValueError("The CSV file must contain 'fish_width' and 'fish_height' columns.")

# Calculate the max_side column
def calculate_max_side(row):
    if row['fish_width'] / row['fish_height'] > 2.2:
        return row['fish_width'] * 0.9
    else:
        return max(row['fish_width'], row['fish_height'])

data['Est_pixel_length'] = data.apply(calculate_max_side, axis=1)

# Save the updated DataFrame back to the CSV file
data.to_csv(file_path, index=False)

print("Updated est. fish length values and saved to the same CSV.")
