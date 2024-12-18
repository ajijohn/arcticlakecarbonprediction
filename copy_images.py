import os
import shutil

# Define the directories
images_dir = "newdata/images"
masks_dir = "newdata/masks"
destination_images_dir = "test/Water Bodies Dataset.1/Images"
destination_masks_dir = "test/Water Bodies Dataset.1/Masks"

# Create the destination directories if they don't exist
os.makedirs(destination_images_dir, exist_ok=True)
os.makedirs(destination_masks_dir, exist_ok=True)

# Get the list of image files
image_files = os.listdir(images_dir)

# Counter for copied images and masks
copied_count = 0

# Iterate over each image file
for image_file in image_files:
    # Check if the corresponding mask file exists
    mask_file = os.path.join(masks_dir, image_file)
    if os.path.exists(mask_file):
        # Copy the image file to the destination images directory
        shutil.copy(os.path.join(images_dir, image_file), destination_images_dir)
        # Copy the mask file to the destination masks directory
        shutil.copy(mask_file, destination_masks_dir)
        copied_count += 1

print(f"Images copied from '{images_dir}' to '{destination_images_dir}'")
print(f"Masks copied from '{masks_dir}' to '{destination_masks_dir}'")
print(f"Count: {copied_count}")
