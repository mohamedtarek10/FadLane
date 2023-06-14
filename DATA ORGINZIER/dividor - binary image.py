import os
import shutil

# Define the source folder containing the images
source_folder = r'// PATH TO YOUR gt_binary_image FILE'

# Define the destination folders for the divided images
destination_folder1 = '//PATH TO SAVE YOUR FIRST CLIENT DATA\\'
destination_folder2 = '//PATH TO SAVE YOUR SECOND CLIENT DATA\\'
destination_folder3 = '//PATH TO SAVE YOUR THIRD CLIENT DATA\\'

# Ensure the destination folders exist
os.makedirs(destination_folder1, exist_ok=True)
os.makedirs(destination_folder2, exist_ok=True)
os.makedirs(destination_folder3, exist_ok=True)

# Loop over the images in the source folder
image_counter = 1
for filename in sorted(os.listdir(source_folder)):
    # Check if the file is an image
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Construct the source and destination paths
        source_path = os.path.join(source_folder, filename)

        # Determine the destination folder based on the image counter
        if image_counter % 3 == 1:
            destination_path = os.path.join(destination_folder1, filename)
        elif image_counter % 3 == 2:
            destination_path = os.path.join(destination_folder2, filename)
        else:
            destination_path = os.path.join(destination_folder3, filename)

        # Copy the image to the destination folder
        shutil.copyfile(source_path, destination_path)

        # Increment the image counter
        image_counter += 1