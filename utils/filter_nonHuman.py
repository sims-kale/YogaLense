import cv2
import mediapipe as mp
import os


def filter_human_images(input_folder, output_folder):
    # Initialize Mediapipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process all images in the input folder
    for dirpath, _, filenames in os.walk(input_folder, output_folder):
        for filename in filenames:
            # Create relative path from dataset_path
            relative_path = os.path.relpath(dirpath, input_folder)
            source_path = os.path.join(dirpath, filename)
            # Create destination directory structure
            dest_dir = os.path.join(output_folder, relative_path)
            os.makedirs(dest_dir, exist_ok=True)
            try:
                if filename.lower().endswith(".jpg" or ".jpeg" or ".png" or ".bmp" or ".gif" or ".tiff" or ".webp"):
                    image = cv2.imread(source_path)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # Run pose estimation
                    results = pose.process(image_rgb)

                # If a human pose is detected, save the image to the output folder
                if results.pose_landmarks:
                    print(f"Human pose detected: {filename} (keeping)")
                    base, ext = os.path.splitext(filename)
                    new_filename = f"{base}.jpg"  # Force JPEG format

                    # Save to correct subdirectory
                    cv2.imwrite(os.path.join(dest_dir, new_filename), image)
                else:
                    print(f"No human pose detected: {filename} (skipping)")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    input_folder = r"./dataset/train/"
    output_folder = "./filtered_data/"
    filter_human_images(input_folder, output_folder)
