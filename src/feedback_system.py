import json
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load
import logging
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore

pose_model_path = r"D:\SHU\Applied ai\Assesment\models\yoga_pose_recognition.keras"
class_indices_path = r"D:\SHU\Applied ai\Assesment\models\class_indices.json"
dataset_path = r"D:\SHU\Applied ai\Assesment\datasets\test"
feedback_model_path = r"D:\SHU\Applied ai\Assesment\models\correction_model_csv.pkl"
csv_path = r"D:\SHU\Applied ai\Assesment\results\test\pose_features_with_labels.csv"
# test_dataset_path = r"D:\SHU\Applied ai\Assesment\datasets\test\train"

# Set up logging
feedback_logger = logging.getLogger("feedback_system")
feedback_handler = logging.FileHandler(r"D:\SHU\Applied ai\Assesment\results\feedback\logs\feedback.log", mode="w")
feedback_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
feedback_logger.addHandler(feedback_handler)
feedback_logger.setLevel(logging.INFO)

# Load model
pose_model = tf.keras.models.load_model(pose_model_path)
feedback_model = load(feedback_model_path)

csv_data = pd.read_csv(csv_path)
# Load class_indices
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)
    #for k in class_indeics.items():
    #    for k in v:
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
    feedback_logger.info(" ")
    feedback_logger.info(f"{image_name}")
    feedback_logger.info(f"Predicted pose-- {predicted_class}")
    # feedback_logger.info(" ")
    return predicted_class

def predict_correctness(image_name):
    feature_row = csv_data[csv_data["Image"] == image_name]
    if feature_row.empty:
        feedback_logger.warning(f"Feature vector not found for image: {image_name}")
        return None, None
    feature_vector = feature_row["Feature_Vector"].values[0]
    feature_vector = feature_vector.replace("nan", "0")  # Replace 'nan' with 0
    feature_vector = np.array(eval(feature_vector))  
    feature_vector = feature_vector.reshape(1, -1) 
    # Predict correctness
    # prediction = feedback_model.predict(feature_vector)
    # correctness = "correct" if prediction[0] == 1 else "incorrect"
    probabilities = feedback_model.predict_proba(feature_vector)  # Get probability scores
    correctness_score = probabilities[0][1] * 100  # Probability of being "correct"
    prediction = feedback_model.predict(feature_vector)  # Predict class (0 or 1)
    correctness = "correct" if prediction[0] == 1 else "incorrect"

    # feedback_logger.info(f"Predicted correctness: {correctness}")
    feedback_logger.info(f"Predicted correctness: {correctness} ({correctness_score:.2f}%)")
    return correctness, correctness_score, feature_row["Label"].values[0]

# Feedback system main func.
def feedback_system(image_path):
    try:
        pose_name = predict_pose(image_path)
        # print(f"Predicted Pose: {pose_name}")
        # Predict correctness
        image_name = os.path.basename(image_path)
        correctness, correctness_score, label = predict_correctness(image_name)

        if correctness is None:
            print("Feature vector not found for this image.")
            return
        print(f"Feedback: Your {pose_name} is {correctness} ({correctness_score:.2f}%).")
        # print(f"Feedback: Your {pose_name} is {correctness}.")
        if correctness == "incorrect":
            print(f"Suggestion: Adjust your pose based on the angles and distances.")
            feedback_logger.info(f"Suggestion: Adjust your pose based on the angles and distances.")

    except Exception as e:
        feedback_logger.error(f"Error in feedback system: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    try:
         for root, dirs, files in os.walk(dataset_path):
            if files: 

                # num_images_to_test = min(3, len(files))  
                # random_images = random.choices(files)  # select random 5 images from each folder

                # for image_name in random_images:
                for image in files:
                    test_image_path = os.path.join(root, image)
                    # print(f"Testing image: {test_image_path}")
                    feedback_system(test_image_path)



    except Exception as e:
        feedback_logger.error(f"Error in main execution: {str(e)}")
        print(f"Error: {str(e)}")