import shutil
import random
from pathlib import Path

# Define the directories
new_images_dir = Path("../newdata/images")
new_json_dir = Path("../newdata/json")
new_masks_dir = Path("../newdata/masks")
train_images_dir = Path("../temp/training/Images")
train_masks_dir = Path("../temp/training/Masks")
test_images_dir = Path("../temp/testing/Images")
test_masks_dir = Path("../temp/testing/Masks")
supplement_images_dir = Path("../temp/Water Bodies Dataset/Images")
supplement_masks_dir = Path("../temp/Water Bodies Dataset/Masks")

test_images_file = Path("../newdata/test_images.txt")

# Number of images to be used for testing
N = 100

# Create the destination directories if they don't exist
train_images_dir.mkdir(parents=True, exist_ok=True)
train_masks_dir.mkdir(parents=True, exist_ok=True)
test_images_dir.mkdir(parents=True, exist_ok=True)
test_masks_dir.mkdir(parents=True, exist_ok=True)

# Delete all files in /testing and /training
for directory in [train_images_dir, train_masks_dir, test_images_dir, test_masks_dir]:
    for file in directory.iterdir():
        if file.is_file():
            file.unlink()

# Get list of non-empty image/mask pairs
json_files = [f for f in new_json_dir.iterdir() if f.is_file()]

# Check if test_images_file exists
if not test_images_file.exists():
    # Randomly pick N non-empty image/mask pairs
    test_images = random.sample(json_files, N)
    # Sort test images by the first number in their name
    test_images.sort(key=lambda x: int(x.stem.split("_")[0]))
    # Write test images to file
    with open(test_images_file, "w") as f:
        for json_file in test_images:
            f.write(f"{json_file.stem}\n")
else:
    # Read test images from file
    with open(test_images_file, "r") as f:
        test_images = [new_json_dir / f"{line.strip()}.json" for line in f]

# Copy test images and masks
for json_file in test_images:
    image_file = new_images_dir / f"{json_file.stem}.jpg"
    mask_file = new_masks_dir / f"{json_file.stem}.jpg"
    shutil.copy(image_file, test_images_dir / image_file.name)
    shutil.copy(mask_file, test_masks_dir / mask_file.name)

# Copy remaining images and masks to training directory
for json_file in json_files:
    if json_file not in test_images:
        image_file = new_images_dir / f"{json_file.stem}.jpg"
        mask_file = new_masks_dir / f"{json_file.stem}.jpg"
        shutil.copy(image_file, train_images_dir / image_file.name)
        shutil.copy(mask_file, train_masks_dir / mask_file.name)

# Copy supplement images and masks to training directory
for image_file in supplement_images_dir.iterdir():
    if image_file.is_file():
        mask_file = supplement_masks_dir / image_file.name
        if mask_file.exists():
            shutil.copy(image_file, train_images_dir / image_file.name)
            shutil.copy(mask_file, train_masks_dir / mask_file.name)
