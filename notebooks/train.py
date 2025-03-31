# %%
import os
import numpy as np
from tqdm import tqdm
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

from keras import Model
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    concatenate,
    Dropout,
)

# %%
image_path = "../temp/training/Images/"
mask_path = "../temp/training/Masks/"
SIZE = 128

# %%
# lists of images and masks names
image_names = sorted(next(os.walk(image_path))[-1])
mask_names = sorted(next(os.walk(mask_path))[-1])

images = np.zeros(shape=(len(image_names), SIZE, SIZE, 3))
masks = np.zeros(shape=(len(image_names), SIZE, SIZE, 1))

for id in tqdm(range(len(image_names)), desc="Images"):
    path = image_path + image_names[id]
    img = np.asarray(Image.open(path)).astype("float") / 255.0
    img = cv.resize(img, (SIZE, SIZE), cv.INTER_AREA)
    images[id] = img

for id in tqdm(range(len(mask_names)), desc="Mask"):
    path = mask_path + mask_names[id]
    mask = np.asarray(Image.open(path)).astype("float") / 255.0
    mask = cv.resize(mask, (SIZE, SIZE), cv.INTER_AREA)
    masks[id] = mask[:, :, :1]

# %%
# Train test split
# images_train, images_test, mask_train, mask_test = train_test_split(images, masks, test_size=0.01)

# Train test split with no images left for testing
images_train = images
images_test = []
mask_train = masks
mask_test = []

print(f"Train images: {len(images_train)}")
print(f"Test images: {len(images_test)}")


# %%
# Data Augmentation
def get_augmentation_generators():
    # Define parameters for image data generator
    data_gen_args = dict(
        rotation_range=15,  # Moderate rotation (lakes can be in any orientation)
        width_shift_range=0.1,  # Small shifts
        height_shift_range=0.1,
        zoom_range=0.1,  # Slight zoom in/out
        horizontal_flip=True,  # Lakes can be flipped
        vertical_flip=True,  # Lakes can be flipped
        fill_mode="reflect",  # Fill with reflected pixels
        brightness_range=[0.9, 1.1],  # Subtle brightness changes
    )

    # Create image generator for images
    image_datagen = ImageDataGenerator(**data_gen_args)
    # Create image generator for masks with the same seed
    mask_datagen = ImageDataGenerator(**data_gen_args)

    return image_datagen, mask_datagen


def augment_data(images, masks, augmentation_factor=3):
    """
    Augment the original dataset by the specified factor.
    Returns augmented images and masks.
    """
    image_datagen, mask_datagen = get_augmentation_generators()

    augmented_images = []
    augmented_masks = []

    print(f"Generating {augmentation_factor}x augmented data...")

    for i in tqdm(range(len(images)), desc="Augmenting"):
        # Get original image and mask
        img = images[i]
        mask = masks[i]

        # Add original image and mask to the augmented datasets
        augmented_images.append(img)
        augmented_masks.append(mask)

        # Generate augmentations
        for j in range(augmentation_factor - 1):
            # Set the same seed for both generators to ensure identical transformations
            seed = np.random.randint(1, 1000)

            # Reshape to match ImageDataGenerator expectations (batch dimension needed)
            img_batch = np.expand_dims(img, 0)
            mask_batch = np.expand_dims(mask, 0)

            # Generate augmented image
            img_gen = image_datagen.flow(img_batch, batch_size=1, seed=seed)
            mask_gen = mask_datagen.flow(mask_batch, batch_size=1, seed=seed)

            # Get the augmented image and mask
            aug_img = next(img_gen)[0]  # [0] to extract from batch
            aug_mask = next(mask_gen)[0]

            augmented_images.append(aug_img)
            augmented_masks.append(aug_mask)

    return np.array(augmented_images), np.array(augmented_masks)


# Apply augmentation to training data
augmented_images_train, augmented_masks_train = augment_data(
    images_train, mask_train, augmentation_factor=3
)

print(f"Original training data size: {len(images_train)}")
print(f"Augmented training data size: {len(augmented_images_train)}")


# Visualize some augmented examples
def visualize_augmentations(
    original_img, original_mask, augmented_imgs, augmented_masks, num_examples=3
):
    plt.figure(figsize=(12, 4 * (num_examples + 1)))

    # Show original
    plt.subplot(num_examples + 1, 2, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(num_examples + 1, 2, 2)
    plt.imshow(original_mask.reshape(SIZE, SIZE), cmap="gray")
    plt.title("Original Mask")
    plt.axis("off")

    # Show augmentations
    for i in range(num_examples):
        idx = i + 1  # Skip the original

        plt.subplot(num_examples + 1, 2, (i + 1) * 2 + 1)
        plt.imshow(augmented_imgs[idx])
        plt.title(f"Augmented Image {i+1}")
        plt.axis("off")

        plt.subplot(num_examples + 1, 2, (i + 1) * 2 + 2)
        plt.imshow(augmented_masks[idx].reshape(SIZE, SIZE), cmap="gray")
        plt.title(f"Augmented Mask {i+1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Visualize a few augmented examples
if len(images_train) > 0:
    idx = np.random.randint(0, len(images_train))
    visualize_augmentations(
        images_train[idx],
        mask_train[idx],
        augmented_images_train,
        augmented_masks_train,
        num_examples=3,
    )


# %%
# Define U-net architecture
def unet_model(input_layer, start_neurons):
    # Contraction path
    conv1 = Conv2D(
        start_neurons, kernel_size=(3, 3), activation="relu", padding="same"
    )(input_layer)
    conv1 = Conv2D(
        start_neurons, kernel_size=(3, 3), activation="relu", padding="same"
    )(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(
        start_neurons * 2, kernel_size=(3, 3), activation="relu", padding="same"
    )(pool1)
    conv2 = Conv2D(
        start_neurons * 2, kernel_size=(3, 3), activation="relu", padding="same"
    )(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(
        start_neurons * 4, kernel_size=(3, 3), activation="relu", padding="same"
    )(pool2)
    conv3 = Conv2D(
        start_neurons * 4, kernel_size=(3, 3), activation="relu", padding="same"
    )(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(
        start_neurons * 8, kernel_size=(3, 3), activation="relu", padding="same"
    )(pool3)
    conv4 = Conv2D(
        start_neurons * 8, kernel_size=(3, 3), activation="relu", padding="same"
    )(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(
        start_neurons * 16, kernel_size=(3, 3), activation="relu", padding="same"
    )(pool4)
    convm = Conv2D(
        start_neurons * 16, kernel_size=(3, 3), activation="relu", padding="same"
    )(convm)

    # Expansive path
    deconv4 = Conv2DTranspose(
        start_neurons * 8, kernel_size=(3, 3), strides=(2, 2), padding="same"
    )(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(
        start_neurons * 8, kernel_size=(3, 3), activation="relu", padding="same"
    )(uconv4)
    uconv4 = Conv2D(
        start_neurons * 8, kernel_size=(3, 3), activation="relu", padding="same"
    )(uconv4)

    deconv3 = Conv2DTranspose(
        start_neurons * 4, kernel_size=(3, 3), strides=(2, 2), padding="same"
    )(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(
        start_neurons * 4, kernel_size=(3, 3), activation="relu", padding="same"
    )(uconv3)
    uconv3 = Conv2D(
        start_neurons * 4, kernel_size=(3, 3), activation="relu", padding="same"
    )(uconv3)

    deconv2 = Conv2DTranspose(
        start_neurons * 2, kernel_size=(3, 3), strides=(2, 2), padding="same"
    )(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(
        start_neurons * 2, kernel_size=(3, 3), activation="relu", padding="same"
    )(uconv2)
    uconv2 = Conv2D(
        start_neurons * 2, kernel_size=(3, 3), activation="relu", padding="same"
    )(uconv2)

    deconv1 = Conv2DTranspose(
        start_neurons * 1, kernel_size=(3, 3), strides=(2, 2), padding="same"
    )(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(
        start_neurons, kernel_size=(3, 3), activation="relu", padding="same"
    )(uconv1)
    uconv1 = Conv2D(
        start_neurons, kernel_size=(3, 3), activation="relu", padding="same"
    )(uconv1)

    # Last conv and output
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    return output_layer


# %%
# Compile unet model
input_layer = Input((SIZE, SIZE, 3))
output_layer = unet_model(input_layer=input_layer, start_neurons=16)

model = Model(input_layer, output_layer)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


# %%
# Post Process
def mask_threshold(image, threshold=0.25):
    return image > threshold


# Callback to show progress of learning on the images after each epoch
class ShowProgress(Callback):
    def __init__(self, save=False):
        self.save = save

    def on_epoch_end(self, epoch, logs=None):
        k = np.random.randint(len(augmented_images_train))
        original_image = augmented_images_train[k][np.newaxis, ...]
        predicted_mask = self.model.predict(original_image).reshape(128, 128)
        proc_mask02 = mask_threshold(predicted_mask, threshold=0.2)
        proc_mask03 = mask_threshold(predicted_mask, threshold=0.3)
        proc_mask04 = mask_threshold(predicted_mask, threshold=0.4)
        proc_mask05 = mask_threshold(predicted_mask, threshold=0.5)
        mask = augmented_masks_train[k].reshape(128, 128)

        plt.figure(figsize=(15, 10))

        plt.subplot(1, 7, 1)
        plt.imshow(original_image[0])
        plt.title("Orginal Image")

        plt.subplot(1, 7, 2)
        plt.imshow(predicted_mask, cmap="gray")
        plt.title("Predicted Mask")

        plt.subplot(1, 7, 3)
        plt.imshow(mask, cmap="gray")
        plt.title("Orginal Mask")

        plt.subplot(1, 7, 4)
        plt.imshow(proc_mask02, cmap="gray")
        plt.title("Processed: 0.2")

        plt.subplot(1, 7, 5)
        plt.imshow(proc_mask03, cmap="gray")
        plt.title("Processed: 0.3")

        plt.subplot(1, 7, 6)
        plt.imshow(proc_mask04, cmap="gray")
        plt.title("Processed: 0.4")

        plt.subplot(1, 7, 6)
        plt.imshow(proc_mask05, cmap="gray")
        plt.title("Processed: 0.5")

        plt.tight_layout()
        plt.show()


# %%
# Training with augmented data
epochs = 100
batch_size = 32

history = model.fit(
    augmented_images_train,  # Use augmented data instead of original
    augmented_masks_train,  # Use augmented masks
    epochs=epochs,
    callbacks=[ShowProgress()],
    batch_size=batch_size,
    shuffle=True,  # Make sure to shuffle
)

# %%
from datetime import datetime

# Get the current date and time
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Save the model with the current date and time in the filename
model.save(f"../models/{current_time}.keras")

# %%
# Make predictions
predictions = model.predict(images_test)
