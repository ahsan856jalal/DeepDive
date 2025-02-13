#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:01:46 2024

@author: ahsanjalal
"""

import pandas as pd
import os
import numpy as np

# Path to depth file
depth_file = "combined_depth.csv"
length_file = "lengths_combined_new.csv"
# Load the depth data
depth_df = pd.read_csv(depth_file)

# Prepare new columns for Range, Length, and pix_length_orig
depth_df['Range'] = None
depth_df['Length'] = None
depth_df['pix_length_orig'] = None
depth_df['Est_pixel_length'] = None

# Function to calculate Euclidean distance
def calculate_pixel_distance(x1, y1, x2, y2):
    max_side = max(abs(x2 - x1), abs(y2 - y1))
    pix_distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return max_side, pix_distance
counter=1
# Iterate through each row in depth_df
for index, row in depth_df.iterrows():
    print(f'{counter}/{len(depth_df)}')
    counter+=1
    # Extract the basename and determine video name
    basename = os.path.basename(row['Filename'])
    if basename.endswith('.png'):
        basename = basename[:-4]  # Remove .png extension
    
    parts = basename.rsplit('.', 1)
    if len(parts) == 2:
        prefix, frame_number = parts[0], parts[1]
    
    # Get the first alphabet for video name
    video_name = prefix[0]
    
    
    # Check if the length file exists
    if not os.path.exists(length_file):
        print(f"Length file not found for {video_name}. Skipping row {index}.")
        continue

    # Load the corresponding length file
    length_df = pd.read_csv(length_file)
    
    # Determine whether it's "L" or "R" and perform the search
    if "_L" in prefix:
        filename_column = 'FilenameLeft'
        frame_column = 'FrameLeft'
    elif "_R" in prefix:
        filename_column = 'FilenameRight'
        frame_column = 'FrameRight'
    else:
        continue  # Skip if neither "L" nor "R" is found
    
    # Search for the matching row in length_df
    match = length_df[
        (length_df[filename_column] == prefix) & 
        (length_df[frame_column] == int(frame_number))
    ]
    
    # If a match is found, extract Range, Length, and calculate pixel distance
    if not match.empty:
        depth_df.at[index, 'Range'] = match.iloc[0]['Range']
        depth_df.at[index, 'Length'] = match.iloc[0]['Length']
        
        # Extract Lx and Ly pairs for pixel distance calculation
        if len(match) > 1:
            Lx1, Ly1 = match.iloc[0]['Lx'], match.iloc[0]['Ly']
            Lx2, Ly2 = match.iloc[1]['Lx'], match.iloc[1]['Ly']
            
            # Calculate pixel distance
            max_side, pixel_distance = calculate_pixel_distance(Lx1, Ly1, Lx2, Ly2)
            depth_df.at[index, 'pix_length_orig'] = pixel_distance
        else:
            print(f"Insufficient data for pixel distance calculation in row {index}.")
    else:
        print(f"No match found in length file for row {index}.")
depth_df = depth_df.sort_values(by='pred_range', ascending=True)
# Save the updated CSV
depth_df.to_csv(depth_file, index=False)
print(f"Updated file saved to {depth_file}")
