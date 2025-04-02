import json
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load
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
    class_indices = {int(v): k for k, v in class_indices.items()} 

# Function to preprocess image for pose recognition
def preprocess_image(image_path, img_height=128, img_width=128):
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Function to predict pose name
def predict_pose(image_path):
    img_array = preprocess_image(image_path)
    predictions = pose_model.predict(img_array)
    predicted_class_index = np.argmax(predictions)  
    predicted_class = class_indices[predicted_class_index]  
    image_name = os.path.basename(image_path)
    feedback_logger.info(f"{image_path}")
    feedback_logger.info(f"{(image_name.split('__')[1])}:Predicted pose-- {predicted_class}")
    feedback_logger.info(" ")
    return predicted_class

# Feedback system
def feedback_system(image_path):
    try:
        # Step 1: Predict pose name
        pose_name = predict_pose(image_path)
        print(f"Predicted Pose: {pose_name}")
        # feedback_logger.info(f"Feedback: Predicted Pose: {pose_name}")

    except Exception as e:
        feedback_logger.error(f"Error in feedback system: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    try:
         for root, dirs, files in os.walk(dataset_path):
            if files:  # Check if the folder contains files
                # Randomly select 2 or 3 images from the current subfolder
                num_images_to_test = min(3, len(files))  # Test up to 3 images or fewer if the folder has less
                random_images = random.sample(files, num_images_to_test)  # Randomly select images

                for image_name in random_images:
                    test_image_path = os.path.join(root, image_name)
                    print(f"Testing image: {test_image_path}")
                    feedback_system(test_image_path)
    except Exception as e:
        feedback_logger.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")