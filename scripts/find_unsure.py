from pathlib import Path

# Define paths
images_dir = Path("../newdata/images")
masks_dir = Path("../newdata/masks")
output_file = Path("../newdata/unsure.txt")

# Get list of image files
image_files = [f.name for f in images_dir.iterdir() if f.is_file()]

# Sort files by the first number in their name
image_files.sort(key=lambda x: int(x.split("_")[0]))

# Check for missing mask files and store them
missing_masks = []
for image_file in image_files:
    mask_path = masks_dir / image_file
    if not mask_path.exists():
        missing_masks.append(image_file)

# Write missing files to text file without the file extension
if missing_masks:
    with open(output_file, "w") as f:
        for filename in missing_masks:
            f.write(f"{Path(filename).stem}\n")
    print(f"Found {len(missing_masks)} images without corresponding masks.")
    print(f"Missing mask filenames have been saved to {output_file}")
else:
    print("All images have corresponding mask files.")
