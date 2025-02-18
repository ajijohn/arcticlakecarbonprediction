import cv2
import json
import numpy as np
from pathlib import Path

images_dir = Path("../newdata/images")
json_dir = Path("../newdata/json")
masks_dir = Path("../newdata/masks")
unsure_file = Path("../newdata/unsure_images.txt")
limit = 865


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

    # read unsure file
    with open(unsure_file, "r") as f:
        unsure_files = set(line.strip() for line in f)

    # sort by number
    image_files = sorted(
        images_dir.rglob("*.*"), key=lambda x: int(x.stem.split("_")[0])
    )

    for image_file in image_files:
        image_number = int(image_file.stem.split("_")[0])
        print(image_number)
        if image_number > limit:
            break
        if image_file.stem not in unsure_files:
            print(f"{image_file} -> {gen_mask_img(image_file)}")
        else:
            print(f"Skipping {image_file} as it is listed in unsure.txt")


if __name__ == "__main__":
    main()
