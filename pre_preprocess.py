from main import preprocessing_logger
import os
import cv2
import logging
import numpy as np
from PIL import Image
import mediapipe as mp
from matplotlib import pyplot as plt
import seaborn as sns
import warnings




def normalize_images(img_resized):
    normalize_array = (
        np.array(img_resized) / 255.0
    )  # Scale pixel values between 0 and 1
    return normalize_array

def preprocess_data(filtered_dataset_path):
    preprocessed_img_files = []  # Store all preprocessed images
    for dirpath, subdir, filenames in os.walk(filtered_dataset_path):
        print("subdir", subdir)
        # if "Tree_pose" in subdir:
        #     continue
        
        processed_images = []  # Reset processed_images for each folder
        for filename in filenames:
            try:
                if filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
                ):
                    file_path = os.path.join(dirpath, filename)
                    preprocessing_logger.info(f"Processing image: {file_path}")
                    try:
                        with Image.open(file_path) as img:  # Open image file

                            if img.size == (0, 0):
                                preprocessing_logger.warning(
                                    f"Skipping {file_path} as it has zero dimensions or is corrupted"
                                )
                                continue  # Skip this file and move to the next

                            if img.mode in ("P", "RGBA", "LA", "I", "F"):
                                img = img.convert("RGB")

                            # Resize image for CNN input
                            cnn_input_size = (224, 224)  # resize for CNN input
                            img_resized = img.resize(cnn_input_size)

                            # Normalize image
                            normalize_img = normalize_images(img_resized)
                            processed_images.append((normalize_img, filename))

                    except Exception as e:
                        preprocessing_logger.error(
                            f"Image is invalid and not processed: {file_path} - {str(e)[:50]}"
                        )
            except Exception as e:
                preprocessing_logger.error(f"Error processing {filename}: {str(e)[:50]}")
                continue  # Skip this file and move to the next

        # Add processed images from the current folder to the main list
        preprocessed_img_files.extend(processed_images)
        preprocessing_logger.info(f"Added {len(processed_images)} images from directory: {dirpath}")

    preprocessing_logger.info("Dataset resized and normalized")
    return preprocessed_img_files
    


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


def extract_keypoints(preprocessed_images):
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks
    mp_drawing_styles = mp.solutions.drawing_styles  # For drawing styles

    all_keypoints = []
    mean_visibility_list = []
    for img, filename in preprocessed_images:
        preprocessing_logger.info(f"Extracting keypoints for {filename}")

        try:
            # Convert the preprocessed image to the correct format
            image = np.array(img * 255, dtype=np.uint8)  # Scale back to 0-255 range
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            image = cv2.resize(image, (255, 255))  # Ensure the image is resized correctly
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert back to RGB for MediaPipe

            # Process the image to detect pose
            results = pose.process(image)

            # Extract keypoints if pose landmarks are detected
            if results.pose_landmarks:
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    keypoints.extend(
                        [landmark.x, landmark.y, landmark.z, landmark.visibility]
                    )
                # Calculate mean visibility for this image
                mean_visibility = np.mean(keypoints[3::4])
                mean_visibility_list.append(mean_visibility)
                preprocessing_logger.info(
                    f"Image {len(mean_visibility_list)}: Mean Visibility = {mean_visibility:.2f}"
                )
                all_keypoints.append((keypoints, filename))

                # Visualize keypoints on the image
                annotated_image = image.copy()
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
                # Convert back to BGR for OpenCV display or saving
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2GRAY)
                base_name = os.path.splitext(filename)[0]  # Get the base name of the image
                output_dir = os.path.join("./results/testing", base_name.split("_")[0])
                os.makedirs(output_dir, exist_ok=True)  # Create subfolder if it doesn't exist
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, annotated_image)
                preprocessing_logger.info(f"Annotated image saved to {output_path}")

        except Exception as e:
            preprocessing_logger.error(f"Error extracting keypoints: {str(e)[:50]}")
            continue
        preprocessing_logger.info(f"Keypoints extracted for {filename}")

    # Visualize the distribution of mean visibility after processing all images
    low_visibility = [v for v in mean_visibility_list if v < 0.6]
    high_visibility = [v for v in mean_visibility_list if v >= 0.6]

    plt.figure(figsize=(12, 6))
    sns.histplot(
        low_visibility,
        bins=50,
        kde=True,
        color="lightgreen",
        alpha=1.0,
        label="Low Visibility (< 0.6)",
    )
    sns.histplot(
        high_visibility,
        bins=50,
        kde=True,
        color="green",
        alpha=1.0,
        label="High Visibility (>= 0.6)",
    )
    plt.axvline(0.6, color="red", linestyle="--", label="Threshold = 0.6")
    plt.title("Distribution of Mean Visibility", fontsize=16)
    plt.xlabel("Mean Visibility", fontsize=14)
    plt.ylabel("Frequency / Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(
        "./results/mean_visibility_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    preprocessing_logger.info("All keypoints extracted and mean visibility calculated.")
    preprocessing_logger.info(f"Total keypoints extracted: {len(all_keypoints)}")
    return all_keypoints, mean_visibility_list
