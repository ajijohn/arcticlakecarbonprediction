import os
import json
import numpy as np
import cv2
from pathlib import Path

root_dir = Path("../newdata/json")
output_dir = Path("../newdata/masks")


def gen_mask_img(json_filename):
    # read json file
    with open(json_filename, "r") as f:
        data = json.load(f)

    original_img_filename = json_filename.parent / data["imagePath"]

    # read image to get shape
    image = cv2.imread(str(original_img_filename))

    # create a blank image
    mask = np.zeros_like(image, dtype=np.uint8)

    # iterate over all shapes and draw them on the mask
    for shape in data["shapes"]:
        points = np.array(
            shape["points"], dtype=np.int32
        )  # tips: points location must be int32
        cv2.fillPoly(mask, [points], (255, 255, 255))

    mask_img_filename = output_dir / f"{json_filename.stem}.jpg"

    # save the mask with the highest quality
    cv2.imwrite(str(mask_img_filename), mask, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    return mask_img_filename


def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    for json_file in root_dir.rglob("*.json"):
        print(f"{json_file} -> {gen_mask_img(json_file)}")


if __name__ == "__main__":
    main()
