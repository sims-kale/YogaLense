from pre_preprocess import preprocess_data, extract_keypoints
import logging

# from train import train_model
# from evaluate import evaluate_model

# DEFINE PATHS
# dataset_path = r"./dataset/train/"
# logging
logging.basicConfig(level=logging.INFO,
                     filename="D:\SHU\Applied ai\Assesment\log_files\PreProcessing_logs\PreProcessing.log", 
                     filemode="w",format='%(levelname)s - %(message)s')
log = logging.getLogger()
filter_dataset_path = r"D:\SHU\Applied ai\Assesment\filtered_data"


def main():
    """Runs the full Yoga Pose Analysis pipeline."""
    # Step 1: Preprocess images (filter non-human images)
    logging.info("Step 1: Preprocessing images...")
    logging.info("-" * int(100))

    # PREPROCESSING - Load and preprocess images
    preprocessed_images = preprocess_data(filter_dataset_path)

    # Step 2: Extract keypoints & save to CSV (MEDIAPIPE)
    logging.info("\nStep 2: Extracting keypoints...")
    all_keypoints = extract_keypoints(preprocessed_images)
    
    # 
    # documentation- https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

    # key_points_dir = "./keypoints/"
    # key_features_csv = "key_feature.csv"
    # key_features(filter_dataset_path,key_points_dir, key_features_csv)

    # # Step 3: Train classification model
    # print("\nStep 3: Training model...")
    # train_model(KEYPOINTS_CSV, MODEL_PATH)
    #
    # # Step 4: Evaluate the trained model
    # print("\nStep 4: Evaluating model performance...")
    # evaluate_model(MODEL_PATH, KEYPOINTS_CSV)

    print("\nYoga Pose Analysis Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
