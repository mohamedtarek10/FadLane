import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# Read each line of the JSON files
json_files = ['// LABELS OF JASON FILE .jason AS A LIST']
folder_name = 'gt_image'  # Folder to save masked images
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

counter = 1  # Counter for naming the images

for json_file in json_files:
    json_gt = open(json_file)
    lines = json_gt.readlines()
    json_gt.close()

    for line in lines:
        gt = json.loads(line)
        gt_lanes = gt['lanes']
        y_samples = gt['h_samples']
        raw_file = os.path.join('//path to your train_set/', gt['raw_file'])
        gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]

        # Load the image
        img = cv2.imread(raw_file)
        img_vis = img.copy()

        # Draw lanes on the image
        for lane in gt_lanes_vis:
            cv2.polylines(img_vis, np.int32([lane]), isClosed=False, color=(255, 255, 255), thickness=5)

        # Create a mask for each lane
        mask = np.zeros_like(img)
        colors = [[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]]

        # Repeat colors cyclically if there are more lanes than colors available
        num_colors = len(colors)
        if len(gt_lanes_vis) > num_colors:
            colors = colors * (len(gt_lanes_vis) // num_colors + 1)

        for i in range(len(gt_lanes_vis)):
            cv2.polylines(mask, np.int32([gt_lanes_vis[i]]), isClosed=False, color=colors[i], thickness=5)

        # Create a grayscale label image
        label = np.zeros((720, 1280), dtype=np.uint8)
        for i in range(len(colors)):
            label[np.where((mask == colors[i]).all(axis=2))] = i + 1

        # Save the label image with incremental name
        label_image_path = os.path.join(folder_name, f"{counter}.png")
        cv2.imwrite(label_image_path, img)
        print("Original Label Path:", raw_file)

        counter += 1  # Increment the counter for the next image