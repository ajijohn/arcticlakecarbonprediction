import os
import json
import numpy as np
import cv2
from pathlib import Path

images_dir = Path("../newdata/images")
json_dir = Path("../newdata/json")
masks_dir = Path("../newdata/masks")
unsure_file = Path("../newdata/unsure.txt")


def gen_mask_img(image_filename):
    json_filename = json_dir / f"{image_filename.stem}.json"
    image = cv2.imread(str(image_filename))

    # create a blank image
    mask = np.zeros_like(image, dtype=np.uint8)

    if json_filename.exists():
        # read json file
        with open(json_filename, "r") as f:
            data = json.load(f)

        # iterate over all shapes and draw them on the mask
        for shape in data["shapes"]:
            points = np.array(
                shape["points"], dtype=np.int32
            )  # tips: points location must be int32
            cv2.fillPoly(mask, [points], (255, 255, 255))

    mask_img_filename = masks_dir / f"{image_filename.stem}.jpg"

    # save the mask with the highest quality
    cv2.imwrite(str(mask_img_filename), mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return mask_img_filename


def main():
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Read unsure file
    with open(unsure_file, "r") as f:
        unsure_files = set(line.strip() for line in f)

    for image_file in images_dir.rglob("*.*"):
        if image_file.name not in unsure_files:
            print(f"{image_file} -> {gen_mask_img(image_file)}")
        else:
            print(f"Skipping {image_file} as it is listed in unsure.txt")


if __name__ == "__main__":
    main()
