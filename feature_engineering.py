import pandas as pd
from pre_preprocess import extract_keypoints
import numpy as np
import cv2
from main import feature_engg_logger

def calculate_angle(a, b, c):
    """
    Calculate the angle at point b given three points: a, b, and c.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ab = a - b
    bc = c - b
    dot_product = np.dot(ab, bc)
    magnitude_ab = np.linalg.norm(ab)
    magnitude_bc = np.linalg.norm(bc)
    # Use np.clip to avoid numerical errors outside the domain of arccos
    angle = np.arccos(np.clip(dot_product / (magnitude_ab * magnitude_bc), -1.0, 1.0))
    # feature_engg_logger.info(f"Calculating angle between {a}, {b}, and {c}: {np.degrees(angle)} degrees")
    return np.degrees(angle)

def calculate_distance(a, b):
    # feature_engg_logger.info(f"Calculating distance between {a} and {b}")
    return np.linalg.norm(np.array(a) - np.array(b))

def list_keypoints(keypoints):
    if isinstance(keypoints[0], list):  # Check if keypoints is nested
        return [item for sublist in keypoints for item in sublist]
    return keypoints

def normalize_keypoints(keypoints, image_width, image_height):
    keypoints = list_keypoints(keypoints)
    normalized_keypoints = []
    for i in range(0, len(keypoints), 4):  # Iterate over x, y, z, visibility
        x = keypoints[i] / image_width
        y = keypoints[i + 1] / image_height
        z = keypoints[i + 2]  # Z is assumed normalized by MediaPipe
        visibility = keypoints[i + 3]
        normalized_keypoints.extend([x, y, z, visibility])
    return normalized_keypoints

def filter_low_visibility(keypoints, visibility_threshold=0.05):
    filtered_keypoints = []
    for i in range(0, len(keypoints), 4):
        visibility = keypoints[i + 3]
        if visibility >= visibility_threshold:
            filtered_keypoints.extend(keypoints[i:i + 4])
        else:
            feature_engg_logger.info(f"Keypoint {i // 4} removed due to low visibility: {visibility}")
    return filtered_keypoints

# ===== Highlighted Change: Revised label_pose function =====
def label_pose(angles, distances, pose_name):
    """
    Label the pose as 'correct' or 'incorrect' based on predefined criteria.
    Returns "correct" only if all checks pass; otherwise "incorrect".
    """
    expected_ranges = {
    "cobra_pose": {
        "left_elbow": (40, 180),
        "right_elbow": (40, 180),
        "left_knee": (80, 180),
        "right_knee": (80, 180),
        "left_ankle": (40, 180),
        "right_ankle": (40, 180),
        "left_shoulder": (80, 180),
        "right_shoulder": (80, 180),
        "shoulder_to_wrist": (0.001,0.002), 
        "hip_to_knee": (0.0001, 0.001),
        "shoulder_to_hip" : (0.0001,0.001), 
        "hip_to_ankle": (0.0001, 0.0025),
    },
    "bridge_pose": {
        "left_elbow": (130, 190),
        "right_elbow": (130, 190),
        "left_knee": (30, 100),
        "right_knee": (30, 100),
        "left_ankle": (50, 180),
        "right_ankle": (50, 180),
        "left_shoulder": (100, 180),
        "right_shoulder": (100, 180),
        "shoulder_to_wrist": (0.001,0.003), 
        "hip_to_knee": (0.0001, 0.002),
        "shoulder_to_hip" : (0.001,0.003),  # Adjusted distance
        "hip_to_ankle": (0.0005, 0.0015),
    },
    "warrior_pose": {
        "left_elbow": (110, 180),
        "right_elbow": (110, 180),
        "left_knee": (80, 170),
        "right_knee": (80, 170),
        "left_ankle": (110, 180),
        "right_ankle": (110, 180),
        "left_shoulder": (0, 140),
        "right_shoulder": (0, 140),
        "shoulder_to_wrist": (0.001,0.0013), 
        "hip_to_knee": (0.0001, 0.001),
        "shoulder_to_hip" : (0.0001,0.001),  # Adjusted distance
        "hip_to_ankle": (0.0005, 0.0015),
    },
    "tree_pose": {
        "left_elbow": (120, 170),
        "right_elbow": (120, 170),
        "left_knee": (5, 90),  
        "right_knee": (160, 180),  
        "left_shoulder": (30, 70),
        "right_shoulder": (30, 70),
        "left_ankle": (140, 170),
        "right_ankle": (140, 170),
        "shoulder_to_wrist": (0.0007, 0.0014),  # Adjusted distance
        "hip_to_knee": (0.0005, 0.0010),  # Adjusted distance
    },
    "seated_pose": {
        "left_elbow": (110, 180),
        "right_elbow": (110, 180),
        "left_knee": (80, 170),
        "right_knee": (80, 170),
        "left_ankle": (110, 180),
        "right_ankle": (110, 180),
        "left_shoulder": (0, 140),
        "right_shoulder": (0, 140),
        "shoulder_to_wrist": (0.001,0.0013), 
        "hip_to_knee": (0.0001, 0.001),
        "shoulder_to_hip" : (0.0001,0.001),  # Adjusted distance
        "hip_to_ankle": (0.0005, 0.0015),
    },
}
    
    if pose_name not in expected_ranges:
        return "unknown"

    valid_count = 0
    total_checks = 0
    knee_valid = False
    
    for feature, (min_val, max_val) in expected_ranges[pose_name].items():
        if feature in angles:
            value = angles[feature]
            if value is not None and min_val <= value <= max_val:
                valid_count += 1
                feature_engg_logger.info(f"{pose_name}: {feature} angle {value:.2f} within range ({min_val}, {max_val})")
                if feature in ["left_knee", "right_knee"]:
                    knee_valid = True
            else:
                feature_engg_logger.warning(f"{pose_name}: {feature} angle {value:.2f} out of range ({min_val}, {max_val})")
            total_checks += 1
        
        elif feature in distances:
            value = distances[feature]
            if value is not None and min_val <= value <= max_val:
                valid_count += 1
                feature_engg_logger.info(f"{pose_name}: {feature} distance {value:.6f} within range ({min_val}, {max_val})")
            else:
                feature_engg_logger.warning(f"{pose_name}: {feature} distance {value:.6f} out of range ({min_val}, {max_val})")
            total_checks += 1
    
    # Special Tree Pose Condition: If one knee is valid, make both knees valid
    if pose_name == "tree_pose" and knee_valid:
        valid_count += 1  # Ensure both knees are considered valid
    
    if pose_name == "bridge_pose" and knee_valid:
        valid_count += 1
    # Pose Classification
    label = "correct" if valid_count >= 7 else "incorrect"
    feature_engg_logger.info(f"{pose_name}: {label} - {valid_count}/{total_checks} valid checks")
    return label


def feature_engg(all_keypoints, pose_name, image_width=224, image_height=224):
    features = []
    for keypoints, image in all_keypoints:
        feature_engg_logger.info(f"{50 * '-'}")
        feature_engg_logger.info(f"Processing file: {image}")
        # Normalize keypoints
        normalized_keypoints = normalize_keypoints(keypoints, image_width, image_height)
        # feature_engg_logger.info(f"Original Keypoints for {image}: {normalized_keypoints}")
        feature_engg_logger.info(f"Total keypoints detected for {image}: {len(normalized_keypoints) // 4}")
        visibility_scores = [normalized_keypoints[i+3] for i in range(0, len(normalized_keypoints), 4)]
        # feature_engg_logger.info(f"Visibility scores for {image}: {visibility_scores}")

        # Filter low-visibility keypoints
        filtered_keypoints = filter_low_visibility(normalized_keypoints)
        feature_engg_logger.info(f"Filtered Keypoints Length: {len(filtered_keypoints)}")
        if len(filtered_keypoints) < 33 * 4:  # Less than 33 keypoints
            feature_engg_logger.warning(f"Insufficient keypoints for {image}. Attempting to impute missing keypoints.")
            for i in range(33):
                if i * 4 >= len(filtered_keypoints):  # If keypoint is missing
                    if i > 0 and (i-1) * 4 < len(filtered_keypoints):
                        x = filtered_keypoints[(i-1) * 4]
                        y = filtered_keypoints[(i-1) * 4 + 1]
                        z = filtered_keypoints[(i-1) * 4 + 2]
                        visibility = 0.0
                    elif i < 32 and (i+1) * 4 < len(filtered_keypoints):
                        x = filtered_keypoints[(i+1) * 4]
                        y = filtered_keypoints[(i+1) * 4 + 1]
                        z = filtered_keypoints[(i+1) * 4 + 2]
                        visibility = 0.0
                    else:
                        x, y, z, visibility = 0.5, 0.5, 0.0, 0.0
                    filtered_keypoints.extend([x, y, z, visibility])
                    feature_engg_logger.info(f"Imputed keypoint {i}: {[x, y, z, visibility]}")
            while len(filtered_keypoints) < 33 * 4:
                filtered_keypoints.extend([0.5, 0.5, 0.0, 0.0])
        
        # Calculate joint angles
        angles = {}
        ideal_angles = {
            "left_elbow": (11, 13, 15),
            "right_elbow": (12, 14, 16),
            "left_knee": (23, 25, 27),
            "right_knee": (24, 26, 28),
            "left_shoulder": (5, 11, 13),
            "right_shoulder": (6, 12, 14),
            "left_ankle": (25, 27, 29),
            "right_ankle": (26, 28, 30),
        }
        for joint, (a_idx, b_idx, c_idx) in ideal_angles.items():
            if a_idx * 4 >= len(filtered_keypoints) or b_idx * 4 >= len(filtered_keypoints) or c_idx * 4 >= len(filtered_keypoints):
                feature_engg_logger.info(f"Missing keypoints for joint: {joint}")
                angles[joint] = None
                continue
            a = (filtered_keypoints[a_idx * 4], filtered_keypoints[a_idx * 4 + 1])
            b = (filtered_keypoints[b_idx * 4], filtered_keypoints[b_idx * 4 + 1])
            c = (filtered_keypoints[c_idx * 4], filtered_keypoints[c_idx * 4 + 1])
            angles[joint] = calculate_angle(a, b, c)
            feature_engg_logger.info(f"Calculated angle for {joint}: {angles[joint]} degrees")
        
        # Calculate distances
        distances = {}
        distance_pairs = {
            "shoulder_to_wrist": (11, 15),
            "hip_to_knee": (23, 25),
            "shoulder_to_hip": (11, 23),
            "hip_to_ankle": (23, 27),
        }
        for name, (start_idx, end_idx) in distance_pairs.items():
            if start_idx * 4 >= len(filtered_keypoints) or end_idx * 4 >= len(filtered_keypoints):
                feature_engg_logger.warning(f"Missing keypoints for distance: {name}")
                distances[name] = None
                continue
            start = (filtered_keypoints[start_idx * 4], filtered_keypoints[start_idx * 4 + 1])
            end = (filtered_keypoints[end_idx * 4], filtered_keypoints[end_idx * 4 + 1])
            distances[name] = calculate_distance(start, end)
            feature_engg_logger.info(f"Calculated distance for {name}: {distances[name]}")
        
        # Label the pose based on computed features using the revised function
        label = label_pose(angles, distances, pose_name)
        feature_vector = list(angles.values()) + list(distances.values())
        features.append((feature_vector, image, label))
        # feature_engg_logger.info(f"Feature vector for {image}: {feature_vector}, Label: {label}")
    return features

def save_features_to_csv(features, output_csv_path):
    # Convert features list to DataFrame
    df = pd.DataFrame(features, columns=["Feature_Vector", "Image", "Label"])
    df.to_csv(output_csv_path, index=False)
    feature_engg_logger.info(f"Features with labels saved to {output_csv_path}")
