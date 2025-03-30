import shutil
import random
from pathlib import Path
import cv2

# Get the directory of the current script
script_dir = Path(__file__).parent

# Define the directories relative to the script directory
new_images_dir = script_dir / "../newdata/images"
new_json_dir = script_dir / "../newdata/json"
new_masks_dir = script_dir / "../newdata/masks"
train_images_dir = script_dir / "../temp/training/Images"
train_masks_dir = script_dir / "../temp/training/Masks"
test_images_dir = script_dir / "../temp/testing/Images"
test_masks_dir = script_dir / "../temp/testing/Masks"
supplement_images_dir = script_dir / "../temp/Water Bodies Dataset/Images"
supplement_masks_dir = script_dir / "../temp/Water Bodies Dataset/Masks"

test_images_file = script_dir / "../newdata/test_images.txt"

# Number of images to be used for testing
test_num = 200

# Image tiling parameters
tile_size = 512
min_tile_size = 512

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


# Function to split an image into tiles
def split_into_tiles(image_path, output_dir):
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return []

    height, width = img.shape[:2]
    base_name = image_path.stem
    ext = image_path.suffix

    tiles = []

    # If the image is already small enough, just copy it directly
    if height <= tile_size and width <= tile_size:
        output_path = output_dir / f"{base_name}{ext}"
        cv2.imwrite(str(output_path), img)
        tiles.append(output_path)
        return tiles

    # Split the image into tiles
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            # Calculate tile dimensions
            h = min(tile_size, height - y)
            w = min(tile_size, width - x)

            # Skip tiles that are too small
            if h < min_tile_size or w < min_tile_size:
                continue

            # Extract the tile
            tile = img[y : y + h, x : x + w]

            # If the tile is not exactly tile_size x tile_size, resize it
            if h != tile_size or w != tile_size:
                tile = cv2.resize(tile, (tile_size, tile_size))

            # Save the tile
            output_path = output_dir / f"{base_name}_x{x}_y{y}{ext}"
            cv2.imwrite(str(output_path), tile)
            tiles.append(output_path)

    return tiles


# Get list of non-empty image/mask pairs
json_files = [f for f in new_json_dir.iterdir() if f.is_file()]

# Check if test_images_file exists
if not test_images_file.exists():
    # Randomly pick N non-empty image/mask pairs
    test_images = random.sample(json_files, test_num)
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

# Copy test images and masks (without splitting)
for json_file in test_images:
    image_file = new_images_dir / f"{json_file.stem}.jpg"
    mask_file = new_masks_dir / f"{json_file.stem}.jpg"

    # Check if files exist before copying
    if image_file.exists():
        shutil.copy(image_file, test_images_dir / image_file.name)
    else:
        print(f"Warning: Test image not found: {image_file}")

    if mask_file.exists():
        shutil.copy(mask_file, test_masks_dir / mask_file.name)
    else:
        print(f"Warning: Test mask not found: {mask_file}")

mask_files = [f for f in new_masks_dir.iterdir() if f.is_file()]
test_image_stems = {json_file.stem for json_file in test_images}

# Copy remaining images and masks to training directory (without splitting)
for mask_file in mask_files:
    if mask_file.stem not in test_image_stems:
        image_file = new_images_dir / f"{mask_file.stem}.jpg"

        # Check if files exist before copying
        if image_file.exists():
            shutil.copy(image_file, train_images_dir / image_file.name)
        else:
            print(f"Warning: Training image not found: {image_file}")

        if mask_file.exists():
            shutil.copy(mask_file, train_masks_dir / mask_file.name)
        else:
            print(f"Warning: Training mask not found: {mask_file}")

# Copy supplement images and masks to training directory WITH splitting
for image_file in supplement_images_dir.iterdir():
    if image_file.is_file():
        mask_file = supplement_masks_dir / image_file.name
        if mask_file.exists():
            # Process image and mask tiles in parallel to maintain correspondence
            image_tiles = split_into_tiles(image_file, train_images_dir)
            mask_tiles = split_into_tiles(mask_file, train_masks_dir)
