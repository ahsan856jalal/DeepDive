import csv
import argparse
import cv2
import glob
import numpy as np
import os
import torch
import matplotlib.cm as cm
from depth_anything_v2.dpt import DepthAnythingV2
import math
import matplotlib.pyplot as plt
def load_yolo_boxes(label_path, img_width, img_height):
    """Load YOLO format bounding boxes from a label file and convert them to pixel coordinates."""
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            boxes.append((x_min, y_min, x_max, y_max))
    return boxes

def normalize_array_to_image(array):
    """
    Normalize a numpy array to the range [0, 255] for saving as an image.
    """
    array_normalized = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX)
    return array_normalized.astype(np.uint8)
def scale_pred_range(pred_range):
    # Apply scaling based on the threshold of 65000
    if 60000 < pred_range < 70000:
        return pred_range / 22
    elif 50000 < pred_range < 60000:
        return pred_range / 24
    elif 40000 < pred_range < 50000:
        return pred_range / 23
    elif 30000 < pred_range < 40000:
        return pred_range / 18
    elif 20000 < pred_range < 30000:
        return pred_range / 15
    elif pred_range > 70000:
        return pred_range / 18
    elif pred_range < 20000:
        return pred_range / 12
    else:
        return pred_range / 20


def calculate_avg_depth(depth, box,image_basename):
    """Calculate the average depth within the given bounding box."""
    x_min, y_min, x_max, y_max = box
    
    box_depth_full = depth[y_min:y_max, x_min:x_max]
    # Calculate intensity range and threshold
    # min_value = np.min(box_depth_full)
    # max_value = np.max(box_depth_full)
    # intensity_range = max_value - min_value
    # threshold = max_value - 0.2 * max_value

    # Filter values above the threshold
    # filtered_array = np.where(box_depth_full > threshold, 0, box_depth_full)
    # avg_depth = np.mean(filtered_array)
    center_y, center_x = box_depth_full.shape[0] // 2, box_depth_full.shape[1] // 2
    circle_radius = int(np.ceil((y_max-y_min)*0.1))
    
    # Create a circular mask
    mask = np.zeros_like(box_depth_full, dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius=circle_radius, color=255, thickness=-1)
    # Calculate the average depth within the circle
    circle_values = box_depth_full[mask == 255]
    avg_depth = np.mean(circle_values)*10# in mm
    avg_depth = scale_pred_range(avg_depth)
    
    # pixel_length=max((y_max-y_min),(x_max-x_min))
    fish_width=abs(x_max-x_min)
    fish_height=abs(y_max-y_min)

    return avg_depth,fish_width,fish_height


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    parser.add_argument('--data-dir', type=str, help='Path to the directory containing images',default="img_data_new")
    parser.add_argument('--out-file', type=str, default='combined_depth.csv', help='Output CSV file to save results')
    parser.add_argument('--input-size', type=int, default=518, help='Input size for depth estimation')
    parser.add_argument('--encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'], help='Model encoder type')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    # Load the DepthAnythingV2 model
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'depth_anything_v2_{args.encoder}.pth', map_location='cpu')) # path to vitb model
    depth_anything = depth_anything.to(DEVICE).eval()
    save_dir='combined_depth'
    os.makedirs(save_dir,exist_ok=True)
    # Get all image filenames
    filenames = glob.glob(os.path.join(args.data_dir, '**/*.png'), recursive=True)

    # Prepare output CSV file
    with open(args.out_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'pred_range','fish_width','fish_height'])

        for k, filename in enumerate(filenames):
            print(f'Processing {k + 1}/{len(filenames)}: {filename}')
            raw_image = cv2.imread(filename)
            if raw_image is None:
                print(f"Error loading image: {filename}")
                continue
            image_basename = os.path.basename(filename) 
            img_height, img_width, _ = raw_image.shape
            label_file = os.path.join("yolov11_overlap", image_basename.replace('.png', '.txt'))

            # Construct the corresponding YOLO label file path
            # label_file = os.path.join(
            #     os.path.dirname(filename),
            #     os.path.splitext(os.path.basename(filename))[0] + '.txt'
            # )
            if not os.path.exists(label_file):
                print(f"Label file not found: {label_file}")
                continue

            # Load YOLO boxes
            boxes = load_yolo_boxes(label_file, img_width, img_height)

            # Perform depth estimation
            depth = depth_anything.infer_image(raw_image, args.input_size)
            filtered_depth_map = cv2.bilateralFilter(depth, d=9, sigmaColor=75, sigmaSpace=75)
            max_depth = np.max(filtered_depth_map)
            zmax=8000#8000
            zmin=1200#1200
            depth1=zmax-filtered_depth_map*((zmax-zmin)/max_depth)
            # depth = (max_depth - depth) # subtract will inverse intensities
            depth_normalized = ((depth1 - depth1.min()) / (depth1.max() - depth1.min()) * 255.0).astype(np.uint8)         
            
            # smoothed_depth = cv2.bilateralFilter(depth1.astype(np.float32), 9, 75, 75)
            
            # Calculate average depth for each box
            for box in boxes:
                avg_depth,fish_width,fish_height = calculate_avg_depth(depth1, box,image_basename)
                save_path = os.path.join(save_dir, f"{os.path.splitext(image_basename)[0]}_depth.png")
                # depth_map_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
                cv2.imwrite(save_path,depth_normalized)
                # Save results to CSV
                writer.writerow([filename,avg_depth,fish_width,fish_height])

    print(f'Average depth values saved to {args.out_file}')
