#!/usr/bin/env python3
"""
Water Bodies Dataset Cleaner

A simple tool to quickly review and remove bad image-mask pairs from the dataset.

Usage:
    python dataset_cleaner.py

Controls:
    'd' or 'D': Delete current image-mask pair
    'q' or 'Q': Quit
    Any other key: Skip to next pair

The script will display each image-mask pair side by side and wait for your input.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import sys


class DatasetCleaner:
    def __init__(self, images_dir, masks_dir):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.deleted_count = 0
        self.current_index = 0
        self.deleted_files = set()  # Track deleted files

        # Get all image files
        self.image_files = sorted([f for f in self.images_dir.glob("*.jpg")])
        self.total_files = len(self.image_files)

        print(f"Found {self.total_files} image files to review")
        print("\nControls:")
        print("  Right Arrow / Space: Next image")
        print("  Left Arrow: Previous image")
        print("  'd' or 'D': Delete current image-mask pair")
        print("  'q' or 'Q': Quit")
        print("\nPress any key to start...")
        cv2.waitKey(0)

    def load_and_resize_image(self, image_path, target_size=(400, 400)):
        """Load and resize image for display"""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        return cv2.resize(img, target_size)

    def load_and_resize_mask(self, mask_path, target_size=(400, 400)):
        """Load and resize mask for display, convert to 3-channel for visualization"""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        mask_resized = cv2.resize(mask, target_size)
        # Convert to 3-channel for better visualization
        mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)
        return mask_colored

    def create_display_image(self, image, mask, filename):
        """Create side-by-side display of image and mask"""
        if image is None or mask is None:
            return None

        # Create a combined image
        combined = np.hstack([image, mask])

        # Add text with filename and progress
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        # Progress info
        current_pos = (self.current_index + 1) if self.current_index is not None else 0
        progress_text = (
            f"{current_pos}/{self.total_files} - Deleted: {self.deleted_count}"
        )
        filename_text = f"File: {filename}"

        # Check if current file is deleted
        status_text = (
            " [DELETED]" if filename in [f.name for f in self.deleted_files] else ""
        )
        filename_text += status_text

        # Add black background for text
        text_bg = np.zeros((80, combined.shape[1], 3), dtype=np.uint8)

        # Add text
        cv2.putText(
            text_bg,
            progress_text,
            (10, 20),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )
        cv2.putText(
            text_bg,
            filename_text,
            (10, 45),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )
        cv2.putText(
            text_bg,
            "Arrow Keys: Navigate | D: Delete | Q: Quit",
            (10, 70),
            font,
            0.5,
            (200, 200, 200),
            thickness,
        )

        # Combine text with image
        final_image = np.vstack([text_bg, combined])

        return final_image

    def delete_pair(self, image_path, mask_path):
        """Delete both image and mask files"""
        try:
            if image_path.exists():
                image_path.unlink()
                print(f"Deleted image: {image_path.name}")

            if mask_path.exists():
                mask_path.unlink()
                print(f"Deleted mask: {mask_path.name}")

            # Track deleted files
            self.deleted_files.add(image_path)
            self.deleted_count += 1
            return True
        except Exception as e:
            print(f"Error deleting files: {e}")
            return False

    def find_next_valid_index(self, start_index, direction=1):
        """Find the next valid (non-deleted) image index"""
        index = start_index
        while 0 <= index < self.total_files:
            image_file = self.image_files[index]
            mask_file = self.masks_dir / image_file.name

            # Check if this file pair still exists (not deleted)
            if image_file.exists() and mask_file.exists():
                return index

            index += direction

        return None

    def run(self):
        """Main cleaning loop with navigation"""
        cv2.namedWindow("Dataset Cleaner", cv2.WINDOW_AUTOSIZE)

        # Start at the first valid image
        self.current_index = self.find_next_valid_index(0)
        if self.current_index is None:
            print("No valid image-mask pairs found!")
            return

        while self.current_index < self.total_files:
            image_file = self.image_files[self.current_index]
            mask_file = self.masks_dir / image_file.name

            # Check if files still exist (skip deleted files)
            if not image_file.exists() or not mask_file.exists():
                # Skip to next valid file
                next_index = self.find_next_valid_index(self.current_index + 1)
                if next_index is not None:
                    self.current_index = next_index
                    continue
                else:
                    print("No more valid files")
                    break

            # Load images
            image = self.load_and_resize_image(image_file)
            mask = self.load_and_resize_mask(mask_file)

            if image is None:
                print(f"Error loading image: {image_file.name}")
                self.current_index += 1
                continue

            if mask is None:
                print(f"Error loading mask: {mask_file.name}")
                self.current_index += 1
                continue

            # Create display
            display_img = self.create_display_image(image, mask, image_file.name)

            if display_img is None:
                self.current_index += 1
                continue

            # Show image
            cv2.imshow("Dataset Cleaner", display_img)

            # Wait for key press
            key = cv2.waitKey(0) & 0xFF

            if key == ord("q") or key == 27:  # Only lowercase q and ESC key
                print(
                    f"\nQuitting... Reviewed {self.current_index + 1}/{self.total_files} files"
                )
                break
            elif key == ord("d"):
                if self.delete_pair(image_file, mask_file):
                    print(f"Deleted pair: {image_file.name}")
                else:
                    print(f"Failed to delete pair: {image_file.name}")
                # Move to next valid file after deletion
                next_index = self.find_next_valid_index(self.current_index + 1)
                if next_index is not None:
                    self.current_index = next_index
                else:
                    print("No more images")
                    break
            elif key == 83 or key == 32:  # Right arrow key or Space
                # Move to next valid file
                next_index = self.find_next_valid_index(self.current_index + 1)
                if next_index is not None:
                    self.current_index = next_index
                else:
                    print("No more images")
            elif key == 81:  # Left arrow key
                # Move to previous valid file
                prev_index = self.find_next_valid_index(
                    self.current_index - 1, direction=-1
                )
                if prev_index is not None:
                    self.current_index = prev_index
                else:
                    print("Already at first image")
            else:
                # Any other key moves forward
                next_index = self.find_next_valid_index(self.current_index + 1)
                if next_index is not None:
                    self.current_index = next_index
                else:
                    print("No more images")

        cv2.destroyAllWindows()

        print(f"\nCleaning complete!")
        print(f"Total files reviewed: {min(self.current_index + 1, self.total_files)}")
        print(f"Total pairs deleted: {self.deleted_count}")
        print(f"Remaining pairs: {self.total_files - self.deleted_count}")


def main():
    # Define paths
    images_dir = "../temp/Water Bodies Dataset/Images"
    masks_dir = "../temp/Water Bodies Dataset/Masks"

    # Check if directories exist
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    if not os.path.exists(masks_dir):
        print(f"Error: Masks directory not found: {masks_dir}")
        sys.exit(1)

    # Create and run cleaner
    cleaner = DatasetCleaner(images_dir, masks_dir)
    cleaner.run()


if __name__ == "__main__":
    main()
