# Training and Testing

## Setup

Create a folder in the project directory called `test`. This folder is ignored in git.

Download `Water Bodies Dataset.zip` from the Google Drive. Unzip the folder and move it to `test`.

Create a copy of the `Water Bodies Dataset` folder, and rename it to `training`. This folder will contain the training data.

```
arcticlakecarbonprediction
└── test
    ├── Water Bodies Dataset
    │   ├── Images
    │   └── Masks
    └── training
        ├── Images
        └── Masks
```

Run `scripts/split_copy_images.py`. This script will copy the test images/masks (specified in `newdata/test_images.txt`) from `newdata` to `test/testing`, and will copy the remaining images/masks to `test/training`. It will create the folders if they don't exist.

```bash
cd scripts
python split_copy_images.py
```

The folder structure should now look like this:

```
arcticlakecarbonprediction
└── test
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

Verify that the correct images exist in all folders. `training` should contain both the old and new images, and `testing` should only contain the new images specified in `newdata/test_images.txt`.

## Training

Run `notebooks/train.ipynb` to train a model. It will save it in the `models` folder.

## Testing

Run `notebooks/test.ipynb` to test a model. You can specify the model path inside the notebook.
