import os
import glob
from PIL import Image
import logging
import numpy as np
import math
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# logging
logging.basicConfig(level=logging.INFO, filename="py_log.log", filemode="w")


def load_data(loaded_dataset_path):
    try:
        # Ensure we only get image files, not folders
        dataset_path = r"./dataset/train/"

        for dirpath, dirnames, filenames in os.walk(dataset_path):
            for filename in filenames:
                # Create relative path from dataset_path
                relative_path = os.path.relpath(dirpath, dataset_path)

                # Create destination directory structure
                dest_dir = os.path.join(loaded_dataset_path, relative_path)
                os.makedirs(dest_dir, exist_ok=True)
                source_path = os.path.join(dirpath, filename)
                try:
                    if filename.endswith(".jpg" or ".jpeg" or ".png" or ".bmp" or ".gif" or ".tiff" or ".webp"):

                        with Image.open(source_path) as img:  # Open image file

                            # Convert image modes properly
                            if img.size == (0, 0):
                                logging.warning(f"Skipping {source_path} as it has zero dimensions or image is corrupted")
                                continue
                            if img.mode in ("P", "RGBA", "LA"):
                                img = img.convert("RGB")

                            # resize image
                            scale_factor = 0.5
                            new_width = max(round(img.width * scale_factor), 1)
                            new_height = max(round(img.height * scale_factor), 1)
                            new_size = (new_width, new_height)
                            img_resized = img.resize(new_size)
                            normalize_array = normalize_imgs(img_resized)  # normalize image

                            normalize_batch = np.expand_dims(normalize_array, axis=0)  # Add batch dimension
                            # # print("img Array", img_array)
                            # agumented_data = next(data_augmentation().flow(normalize_array, batch_size=1))  # data augmentation
                            # print(" agumented_data" + str(agumented_data.shape))

                            normalize_img = np.squeeze(normalize_batch, axis=0)  # Remove batch dimension
                            normalize_img = Image.fromarray((normalize_img * 255).astype(np.uint8)) # Convert to image

                            # Create new filename
                            base, ext = os.path.splitext(filename)
                            new_filename = f"{base}_resized.jpg"  # Force JPEG format

                            # Save to correct subdirectory
                            dest_path = os.path.join(dest_dir, new_filename)
                            normalize_img.save(dest_path)
                            print(f"Saved: {dest_path}")
                    else:
                        logging.warning(f"Skipping {source_path} as it is not an image file")
                except Exception as e:
                    logging.error(f"Error processing {source_path}: {str(e)}")

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")


def normalize_imgs(image):
    return np.array(image) / 255.0  # Scale pixel values between 0 and 1


# def data_augmentation():
#     return ImageDataGenerator(
#         rotation_range=20,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest'
#     )
#      # datagen



def main():
    loaded_dataset_path = "D:/SHU/Applied ai/Assesment/loaded_dataset/train/"
    load_data(loaded_dataset_path)


main()
