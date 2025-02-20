# Documentation

## Labeling

### Setup

Install Labelme:

```bash
pip install labelme
```

Run Labelme:

```bash
labelme
```

Click "Open Dir" and select the `newdata/images` directory. You should now see the images and a file list.

Click "File" > "Change Output Dir" and select the `newdata/json` directory. The entries in the file list
with checkmarks are images that have been labeled.

![image](https://github.com/user-attachments/assets/67e6c76a-101a-4e8c-aa01-4f260d202ce5)

Use the file list or "Prev Image"/"Next Image" to navigate to an unlabeled image.

### AI Polygons

Make sure the "AI Model" is set to "EfficientSam (accuracy)". Click "Edit" > "Create AI-Polygon" to begin creating an AI polygon.

Now click in the middle of a lake. Try to aim for the darkest area so that the AI is less sensitive. Move your mouse around until the outline looks good, then click again. You may need to repeat this a few times. You can use Ctrl+Z to undo.

When you are finished with a polygon, press Enter. In the popup, just put in "1" and press Enter.

![image](https://github.com/user-attachments/assets/ad222d37-549f-496b-8387-e9f88bd5554c)

### Tips

Sometimes the AI polygon will make some mistakes. You can use "Edit Polygons" to drag any of the nodes. Shift+click a node to remove it, and click on any edge to create a new node.

If the AI polygon does not work well, or if you are trying to label a very small lake, it can be easier to use the manual tool (click "Create Polygons").

For larger, more oddly shaped lakes, it can often be effective to create multiple overlapping polygons to cover the whole area.

If it is hard to tell whether something is a lake or not, viewing the [full image](https://drive.google.com/file/d/1adtjKAnc-Lfhgf7AT6I-UqnRaY8zfYPp/view) may help with providing a better view, since it is sharper and you can see the surrounding context. The last two number of the name of the tile denote its coordinates in the full image.

### Saving

Once all the lakes in an image are labeled, press Ctrl+S. In the "Choose File" window, just press Enter to save.

### Skipping

If you are sure an image has no lakes, skip it and do not save any masks. This will result in an empty mask being generated later on by the conversion script.

If you are unsure about an image and want to exclude it from the data, add the image name to the `unsure.txt` file. This will prevent any masks from being generated for that image.

### Converting

To convert the Labelme JSON files to mask images, run `scripts/json_to_mask.py`.

Make sure to set the`limit` variable to the current number of lakes (inclusive) that have been labeled.

```bash
python scripts/json_to_mask.py
```

## Training

### Setup

Create a folder in the project directory called `temp`. This folder is ignored in git.

Download `Water Bodies Dataset.zip` from the Google Drive. Unzip the folder and move it to `temp`. The folder structure should look like this:

```
arcticlakecarbonprediction
└── temp
    └── Water Bodies Dataset
        ├── Images
        └── Masks
```

Run `scripts/split_copy_images.py`. This script will perform the following actions:

1. Create the folders `temp/training` and `temp/testing` if they don't exist and delete all files in them.
2. Copy the newdata test images/masks (specified in `newdata/test_images.txt`) from `newdata` to `temp/testing`
3. Copy the remaining newdata images/masks to `temp/training`.
4. Copy the `Water Bodies Dataset` images/masks to `temp/training`.

```bash
python scripts/split_copy_images.py
```

The folder structure should now look like this:

```
arcticlakecarbonprediction
└── temp
    ├── Water Bodies Dataset
    │   ├── Images
    │   └── Masks
    ├── training
    │   ├── Images
    │   └── Masks
    └── testing
        ├── Images
        └── Masks
```

Verify that the correct images exist in all folders. `temp/training` should contain both the old and new images, and `temp/testing` should only contain the test images specified in `newdata/test_images.txt`.

### Updating Data

If there are any updates to the new data, the training and testing folders should be updated. Simply run the `scripts/split_copy_images.py` script again.

### Actually Training

Run `notebooks/train.ipynb` to train a model. It will save it in the `models` folder.

## Testing

Run `notebooks/test.ipynb` to test a model. You will need to specify the correct model path inside the notebook.
