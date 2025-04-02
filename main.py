import os
import warnings

import logging

# Suppress warnings
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings(
    "ignore", category=UserWarning, module="PIL"
)  # Suppress PIL warnings

flitered_dataset_path = "/filtered_data"

# Main logger
# Preprocessing logger
preprocessing_logger = logging.getLogger("preprocessing")
preprocessing_handler = logging.FileHandler("./Final_results/logs/preprocessing.log", mode="w")
preprocessing_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
preprocessing_logger.addHandler(preprocessing_handler)
preprocessing_logger.setLevel(logging.INFO)

# Feature Engineering logger
feature_engg_logger = logging.getLogger("feature_engg")
# feature_engg_handler = logging.FileHandler("./log_files/feature_engg.log", mode="w")
feature_engg_handler = logging.FileHandler("./Final_results/logs/feature_engg.log", mode="w")
feature_engg_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
feature_engg_logger.addHandler(feature_engg_handler)
feature_engg_logger.setLevel(logging.INFO)


def main():
    
    from pre_preprocess import preprocess_data, extract_keypoints
    from feature_engineering import feature_engg, save_features_to_csv

    try:
        # Step 1: Preprocessing images
        preprocessing_logger.info("Step 1: Preprocessing images...")
        preprocessed_img_files = preprocess_data(flitered_dataset_path)
                    
                 
        preprocessing_logger.info("Image preprocessing completed.")
        preprocessing_logger.info("-" * 100)

        # Step 2: Extract keypoints & save to CSV (MEDIAPIPE)
        preprocessing_logger.info("Step 2: Extracting keypoints...")
        all_keypoints, mean_visibility_list = extract_keypoints(preprocessed_img_files)
        preprocessing_logger.info(f"Total keypoints extracted: {len(all_keypoints)}")
        preprocessing_logger.info("Keypoint extraction completed.")
        preprocessing_logger.info("-" * 100)

        # Split keypoints by pose
        cobra_keypoints = [kp for kp in all_keypoints if "cobra" in kp[1].lower()]
        bridge_keypoints = [kp for kp in all_keypoints if "bridge" in kp[1].lower()]
        warrior_keypoints = [kp for kp in all_keypoints if "warrior" in kp[1].lower()]
        tree_keypoints = [kp for kp in all_keypoints if "tree" in kp[1].lower()]
        seated_keypoints = [kp for kp in all_keypoints if "seated" in kp[1].lower()]

        # Step 3: Feature Engineering
        feature_engg_logger.info("Step 3: Feature Engineering...")
        # Define pose names and their corresponding keypoints
        pose_keypoints = {
            "cobra_pose": cobra_keypoints,
            "bridge_pose": bridge_keypoints,
            "warrior_pose": warrior_keypoints,
            "tree_pose": tree_keypoints,
            "seated_pose": seated_keypoints,
        }

        # Extract features and labels for each pose
        all_features = []
        for pose_name, keypoints in pose_keypoints.items():
            features = feature_engg(keypoints, pose_name)
            all_features.extend(features)

        feature_engg_logger.info("Feature engineering completed.")
        feature_engg_logger.info("-" * 100)

        # Save features to CSV
        feature_engg_logger.info("Saving features to CSV...")
        output_csv_path = "./Final_results/extracted_features_with_labels.csv"
        save_features_to_csv(all_features, output_csv_path)

        print("\nYoga Pose Analysis Pipeline Completed Successfully!")

    except Exception as e:
        preprocessing_logger.error(f"Pipeline failed: {str(e)}")
        feature_engg_logger.error(f"Pipeline failed: {str(e)}")


if __name__ == "__main__":
    main()
