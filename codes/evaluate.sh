#!/bin/bash

# Set the folder path
folder_path="DeepDive/codes"

# Move to the folder
cd "$folder_path" || { echo "Error: Could not change directory to $folder_path"; exit 1; }
python step10_calc_evaluation_metrices.py