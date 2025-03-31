import cv2
import mediapipe as mp
import os

# Suppress unnecessary logs
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def filter_human_images(input_folder, human_folder, non_human_folder):
    print("Starting to filter human images...")
    
    # Initialize Mediapipe Pose model
    mp_pose = mp.solutions.pose
    with mp_pose.Pose() as pose:
        print(mp_pose)
        # Create output folder if it doesn't exist
        os.makedirs(human_folder, exist_ok=True)
        os.makedirs(non_human_folder, exist_ok=True)
        # print(human_folder)

        # Walk through the input folder
        for dirpath, _, filenames in os.walk(input_folder):
            print(dirpath)
            print(filenames)
            # Create a relative path for destination structure
            relative_path = os.path.relpath(dirpath, input_folder)
            dest_dir1= os.path.join(human_folder, relative_path)
            dest_dir2 = os.path.join(non_human_folder, relative_path)
            os.makedirs(dest_dir1, exist_ok=True)
            os.makedirs(dest_dir2, exist_ok=True)


            for filename in filenames:
                source_path = os.path.join(dirpath, filename)
                try:
                    # Check for valid image file formats
                    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp")):
                        print(f"Processing file: {filename}")
                        image = cv2.imread(source_path)
                        if image is None:
                            print(f"Unable to load image: {filename}")
                            continue

                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Run pose estimation
                        results = pose.process(image_rgb)

                        # If a human pose is detected, save the image
                        if results.pose_landmarks:
                            print(f"Human pose detected: {filename} (keeping)")
                            base, _ = os.path.splitext(filename)
                            new_filename = f"{base}.jpg"  # Force JPEG format
                            cv2.imwrite(os.path.join(dest_dir1, new_filename), image)
                        else:
                            print(f"No human pose detected: {filename} (skipping)")
                            base, _ = os.path.splitext(filename)
                            new_filename = f"{base}.jpg"  # Force JPEG format
                            cv2.imwrite(os.path.join(dest_dir2, new_filename), image)
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    input_folder = r"D:\SHU\Applied ai\Assesment\dataset"
    human_folder = "./filtered_data/"
    non_human_folder = "./non_filtered_data"
    filter_human_images(input_folder, human_folder, non_human_folder)