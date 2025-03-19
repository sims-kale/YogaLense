import os
import cv2
import logging
import numpy as np
import pandas as pd
from PIL import Image
import mediapipe as mp
import tensorflow as tf
from matplotlib import pyplot as plt
from mediapipe import solutions

# logging
logging.basicConfig(level=logging.INFO, filename="py_log.log", filemode="w")


# Validate the dataset
def preprocess_data(dataset_path, flitered_dataset_path):
    try:
        for dirpath, _, filenames in os.walk(dataset_path):
            for filename in filenames:
                # Create relative path from dataset_path
                relative_path = os.path.relpath(dirpath, dataset_path)
                source_path = os.path.join(dirpath, filename)
                try:
                    if filename.lower().endswith(".jpg" or ".jpeg" or ".png" or ".bmp" or ".gif" or ".tiff" or ".webp"):
                        with Image.open(source_path) as img:  # Open image file

                            # Convert image modes properly
                            if img.size == (0, 0):
                                logging.warning(
                                    f"Skipping {source_path} as it has zero dimensions or image is corrupted")
                                continue

                            if img.mode in ("P", "RGBA", "LA", "I", "F"):
                                img = img.convert("RGB")

                            # resize image
                            scale_factor = 0.5
                            new_width = max(round(img.width * scale_factor), 1)
                            new_height = max(round(img.height * scale_factor), 1)
                            new_size = (new_width, new_height)
                            img_resized = img.resize(new_size)

                            # Normalize image
                            normalize_img = normalize_images(img_resized)

                            # data augmentation
                            # augmented_data = next(data_augmentation().flow(normalize_array, batch_size=1))
                            # print(" augmented_data" + str(augmented_data.shape))
                            save_preprocess_images(flitered_dataset_path, filename, relative_path,
                                                   normalize_img)

                    else:
                        logging.warning(f"Skipping {source_path} as it is not an image file")
                except Exception as e:
                    logging.error(f"Error processing {source_path}: {str(e)}")

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")


def normalize_images(img_resized):
    normalize_array = np.array(img_resized) / 255.0  # Scale pixel values between 0 and 1
    normalize_batch = np.expand_dims(normalize_array, axis=0)  # Add batch dimension
    normalize_img = np.squeeze(normalize_batch, axis=0)  # Remove batch dimension
    normalize_img = Image.fromarray((normalize_img * 255).astype(np.uint8))  # Convert to image
    return normalize_img


def save_preprocess_images(flitered_dataset_path, filename, relative_path, normalize_img):
    # Create destination directory structure
    dest_dir = os.path.join(flitered_dataset_path, relative_path)
    os.makedirs(dest_dir, exist_ok=True)

    # Create new filename
    base, ext = os.path.splitext(filename)
    new_filename = f"{base}_resized.jpg"  # Force JPEG format

    # Save to correct subdirectory
    dest_path = os.path.join(dest_dir, new_filename)
    normalize_img.save(dest_path)
    print(f"Saved: {dest_path}")


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


def key_features(flitered_dataset_path, key_features_csv):
    # print("test")

    # Load MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()  # 33 key-points
    mp_drawing = mp.solutions.drawing_utils

    # Load image
    for dirpath, dirnames, filenames in os.walk(flitered_dataset_path):
        # Check if the current directory is "Tree_Pose_or_Vrksasana__images"
        if os.path.basename(dirpath) == "Tree_Pose_or_Vrksasana__images":
            print(f"Found directory: {dirpath}")

            for index, img in enumerate(filenames):
                if index == 1:  # Process only the first 3 images
                    break

                # Create the full path to the image
                image_path = os.path.join(dirpath, img)
                print(f"Processing image: {image_path}")

                # Load the image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error: Unable to load image {image_path}")
                    continue  # Skip to the next image if loading fails

                print("Image loaded successfully.")

                try:
                    # Convert the image to RGB
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Process image to detect pose
                    results = pose.process(img_rgb)
                    print("Pose detection results:", results)

                    # Extract keypoints if pose detected
                    if results.pose_landmarks:
                        keypoints = []
                        for i, landmark in enumerate(results.pose_landmarks.landmark):
                            keypoints.append([i, landmark.x, landmark.y, landmark.z, landmark.visibility])

                        # Convert keypoints to a DataFrame
                        df = pd.DataFrame(keypoints, columns=["Joint", "X", "Y", "Z", "Visibility"])

                        # Save keypoints to CSV file
                        csv_filename = os.path.splitext(os.path.basename(image_path))[0] + key_features_csv
                        df.to_csv(csv_filename, index=False)

                        print(f"Keypoints saved to {csv_filename}")

                    # Optionally, draw the pose landmarks on the image
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                         circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2,
                                                                           circle_radius=5))

                    # Display or save the processed image
                    # cv2.imshow("Processed Image", image)
                    # cv2.waitKey(0)  # Wait for a key press to close the window
                    # cv2.destroyAllWindows()

                    f, axes = plt.subplots(1, 1, figsize=(10, 10))
                    axes.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    axes.title.set_text("33 Key-points Detected")
                    plt.show()

                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
