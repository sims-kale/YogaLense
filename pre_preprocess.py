import os
import cv2
import logging
import numpy as np
from PIL import Image
import mediapipe as mp
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings(
    "ignore", category=UserWarning, module="PIL"
)  # Suppress PIL warnings


def normalize_images(img_resized):
    normalize_array = (
        np.array(img_resized) / 255.0
    )  # Scale pixel values between 0 and 1
    return normalize_array


def preprocess_data(flitered_dataset_path):
    try:
        logging.info(f"Starting Pre-Processing....")
        processed_images = []
        for dirpath, _, filenames in os.walk(flitered_dataset_path):
            for filename in filenames[0:500]:
                try:
                    if filename.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")
                    ):
                        file_path = os.path.join(dirpath, filename)
                        try:
                            with Image.open(file_path) as img:  # Open image file

                                if img.size == (0, 0):
                                    logging.warning(
                                        f"Skipping {file_path} as it has zero dimensions or image is corrupted"
                                    )
                                    continue

                                if img.mode in ("P", "RGBA", "LA", "I", "F"):
                                    img = img.convert("RGB")

                                # Resize image for CNN input
                                cnn_input_size = (224, 224)  # resize for CNN input
                                img_resized = img.resize(cnn_input_size)

                                # Normalize image
                                normalize_img = normalize_images(img_resized)
                                # logging.info(f"Image processed and normalized: {file_path}")
                                # logging.info("-" * int(100))
                                processed_images.append(normalize_img)

                        except Exception as e:
                            logging.error(
                                f"Image is invalid not processed: {file_path}"
                            )
                except Exception as e:
                    logging.error(f"Error processing {filename}: {str(e)[:50]}")
                    continue

                    # data augmentation
                    # augmented_data = next(data_augmentation().flow(normalize_array, batch_size=1))
                    # print(" augmented_data" + str(augmented_data.shape))
                    # save_preprocess_images(flitered_dataset_path, filename, relative_path,
                    #    normalize_img)
        logging.info(f"Dataset resized and normalized")
        return processed_images
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)[:50]}")


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
    print("test")

    all_keypoints = []
    mean_visibility_list = []
    for img in preprocessed_images:

        try:
            # Convert the preprocessed image to the correct format
            image = np.array(img * 255, dtype=np.uint8)  # Scale back to 0-255 range
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image to detect pose
            results = pose.process(image)

            # Extract keypoints if pose landmarks are detected
            if results.pose_landmarks:
                keypoints = []
                for landmark in results.pose_landmarks.landmark:
                    keypoints.extend(
                        [landmark.x, landmark.y, landmark.z, landmark.visibility]
                    )
                    # logging.info(f"kepoints exteded for the image", keypoints)
                # Calculate mean visibility for this image
                # print("keypoints", keypoints)
                mean_visibility = np.mean(keypoints[3::4])
                mean_visibility_list.append(mean_visibility)
                # Print the mean visibility for each image
                logging.info(
                    f"Image {len(mean_visibility_list)}: Mean Visibility = {mean_visibility:.2f}"
                )
                all_keypoints.append(keypoints)
                logging.info("All Keypoints are extracted in the list")
        except Exception as e:
            logging.error(f"Error extracting keypoints: {str(e)[:50]}")
            continue

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

    logging.info("All keypoints extracted and mean visibility calculated.")
    return all_keypoints, mean_visibility_list
