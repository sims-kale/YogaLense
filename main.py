from pre_preprocess import preprocess_data, key_features

# from train import train_model
# from evaluate import evaluate_model

# DEFINE PATHS
dataset_path = r"./dataset/train/"
loaded_dataset_path = r"D:\SHU\Applied ai\Assesment\filtered_data"


def main():
    """Runs the full Yoga Pose Analysis pipeline."""
    print("Starting Yoga Pose Analysis Pipeline...\n")

    # Step 1: Preprocess images (filter non-human images)
    print("Step 1: Preprocessing images...")

    # PREPROCESSING - Load and preprocess images
    # preprocess_data(dataset_path, loaded_dataset_path)

    # Step 2: Extract keypoints & save to CSV (MEDIAPIPE)
    print("\nStep 2: Extracting keypoints...")
    # documentation- https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
    key_features_csv = "key_feature.csv"
    key_features(loaded_dataset_path, key_features_csv)

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
