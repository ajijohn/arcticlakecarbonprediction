import os
import numpy as np
from tqdm import tqdm
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt

image_path = "../temp/sentinel2_visual.tif"
output_dir = "../newdata/images"
SIZE = 128
COUNT = 2000

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the image
image = Image.open(image_path)
image = image.convert("RGB")  # Ensure image is in RGB format
image_np = np.array(image)

# Get image dimensions
height, width, _ = image_np.shape

# Calculate the number of tiles in each dimension
num_tiles_x = width // SIZE
num_tiles_y = height // SIZE

# Initialize tile counter
tile_count = 0

# Loop through the image and save tiles
for i in range(num_tiles_y):
    for j in range(num_tiles_x):
        if tile_count >= COUNT:
            break
        # Calculate the coordinates of the top left corner
        x = j * SIZE
        y = i * SIZE
        # Extract the tile
        tile = image_np[y : y + SIZE, x : x + SIZE]
        # Save the tile
        tile_image = Image.fromarray(tile)
        tile_image.save(
            os.path.join(output_dir, f"{tile_count}_{x}_{y}.jpg"), quality=100
        )
        tile_count += 1

print(f"Saved {tile_count} tiles in {output_dir}")
