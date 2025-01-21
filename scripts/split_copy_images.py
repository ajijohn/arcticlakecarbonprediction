import os
import shutil
import random

new_images_dir = "../newdata/images"
new_masks_dir = "../newdata/masks"
train_images_dir = "../test/training/Images"
train_masks_dir = "../test/training/Masks"
test_images_dir = "../test/testing/Images"
test_masks_dir = "../test/testing/Masks"

# Create the destination directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_masks_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_masks_dir, exist_ok=True)

# Get the list of new image files
image_files = os.listdir(new_images_dir)

# Filter out images that do not have corresponding masks
image_files = [f for f in image_files if os.path.exists(os.path.join(new_masks_dir, f))]

# Shuffle the list to ensure randomness
random.shuffle(image_files)

# Define the number of images to use for testing
N = 100  # You can change this value as needed

# Split the image files into test and train sets
test_image_files = image_files[:N]
train_image_files = image_files[N:]


# Function to copy files
def copy_files(
    image_files, src_images_dir, src_masks_dir, dest_images_dir, dest_masks_dir
):
    for image_file in image_files:
        mask_file = image_file  # Mask file has the same name as the image file
        shutil.copy(os.path.join(src_images_dir, image_file), dest_images_dir)
        shutil.copy(os.path.join(src_masks_dir, mask_file), dest_masks_dir)


# Copy test files
copy_files(
    test_image_files, new_images_dir, new_masks_dir, test_images_dir, test_masks_dir
)

# Copy train files
copy_files(
    train_image_files, new_images_dir, new_masks_dir, train_images_dir, train_masks_dir
)

print(f"Copied {len(test_image_files)} files to testing directories.")
print(f"Copied {len(train_image_files)} files to training directories.")
