#!/usr/bin/env python3

"""
Created on Tue Dec 10 11:53:37 2024

@author: ahsanjalal
"""

import cv2
import os

# Paths to the annotated directory
image_dir = "img_data_new"
annot_dir = "img_data_new" # Same directory for text files

# Get list of PNG files
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]

# Function to parse YOLO format annotations
def parse_yolo_annotations(txt_file, img_width, img_height):
    boxes = []
    with open(txt_file, 'r') as f:
        for line in f:
            # YOLO format: class x_center y_center width height
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x_center, y_center, width, height = map(float, parts)
            # Convert YOLO normalized values to pixel coordinates
            x_min = int((x_center - width / 2) * img_width)
            x_max = int((x_center + width / 2) * img_width)
            y_min = int((y_center - height / 2) * img_height)
            y_max = int((y_center + height / 2) * img_height)
            boxes.append((x_min, y_min, x_max, y_max))
    return boxes

# Loop through images
total_images = len(image_files)
current_index = 0

for image_file in image_files:
    current_index += 1
    
    # Read image
    img_path = os.path.join(image_dir, image_file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load image: {image_file}")
        continue
    
    img_height, img_width = image.shape[:2]

    # Read corresponding annotation file
    txt_file = os.path.join(annot_dir, image_file.replace('.png', '.txt'))
    if not os.path.exists(txt_file):
        print(f"No annotation file found for {image_file}")
        continue

    # Parse YOLO annotations and draw boxes
    boxes = parse_yolo_annotations(txt_file, img_width, img_height)
    for (x_min, y_min, x_max, y_max) in boxes:
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Add image name and count to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 0, 0)
    thickness = 2
    text_position = (10, 30)  # Position at top-left corner
    count_position = (10, 60)  # Position below the filename text
    cv2.putText(image, image_file, text_position, font, font_scale, font_color, thickness)
    cv2.putText(image, f"{current_index}/{total_images}", count_position, font, font_scale, font_color, thickness)

    # Display the image
    cv2.imshow('Annotated Image', image)

    # Wait for a key press
    key = cv2.waitKey(0)
    if key == ord('q'):  # Quit the program
        break
    elif key == ord('d'):  # Keep the image and text file
        print(f"Kept: {image_file} and corresponding annotation.")
        continue
    elif key == ord('a'):  # Delete the image and text file
        os.remove(img_path)  # Delete image
        if os.path.exists(txt_file):  # Delete annotation file if it exists
            os.remove(txt_file)
        print(f"Deleted: {image_file} and corresponding annotation.")
        continue

# Close all windows
cv2.destroyAllWindows()
