import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load  # Corrected import for joblib
import logging
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

pose_model_path = "./models/yoga_pose_recognition.keras"
class_indices_path = r"D:\SHU\Applied ai\Assesment\models\class_indices.json"
dataset_path = r"D:\SHU\Applied ai\Assesment\testing_dataset"

# Set up logging
feedback_logger = logging.getLogger("feedback_system")
feedback_handler = logging.FileHandler("./Final_results/logs/feedback_system.log", mode="w")
feedback_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
feedback_logger.addHandler(feedback_handler)
feedback_logger.setLevel(logging.INFO)

# Load the pretrained CNN model
pose_model = tf.keras.models.load_model(pose_model_path)

# Load class_indices
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)
    class_indices = {int(v): k for k, v in class_indices.items()}  # Reverse the mapping

# Function to preprocess image for pose recognition
def preprocess_image(image_path, img_height=128, img_width=128):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    return np.expand_dims(img_array, axis=0)

# Function to predict pose name
def predict_pose(image_path):
    img_array = preprocess_image(image_path)
    predictions = pose_model.predict(img_array)
    predicted_class_index = np.argmax(predictions)  # Get the index of the highest probability
    predicted_class = class_indices[predicted_class_index]  # Map index to class name
    feedback_logger.info(f"Predicted pose for {image_path}: {predicted_class}")
    return predicted_class

# Feedback system
def feedback_system(image_path):
    try:
        # Step 1: Predict pose name
        pose_name = predict_pose(image_path)
        print(f"Predicted Pose: {pose_name}")
        feedback_logger.info(f"Feedback: Predicted Pose: {pose_name}")

    except Exception as e:
        feedback_logger.error(f"Error in feedback system: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    try:
        # Test the feedback system with an example image
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                test_image_path = os.path.join(root, file)
                if os.path.isfile(test_image_path):
                    print(f"Testing image: {test_image_path}")
                    feedback_system(test_image_path)
    except Exception as e:
        feedback_logger.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")