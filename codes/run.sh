#!/bin/bash

# Set the folder path
folder_path="DeepDive/codes"

# Move to the folder
cd "$folder_path" || { echo "Error: Could not change directory to $folder_path"; exit 1; }



python step1b_remove_small_fish_images_from_dir.py
python step3_filter_range_save.py
python step4_yolov11_inferences.py
python step5_save_overalap_predictions_with_GT.py
python step6_calc_depth_folder.py
python step7_save_range_length_to_csv.py
python step8_pixel_length.py
python step9_calc_length.py
python step10_calc_evaluation_metrices.py