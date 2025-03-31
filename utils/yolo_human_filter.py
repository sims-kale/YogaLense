from ultralytics import YOLO
import cv2
import os
import logging

logfile = r"D:\SHU\Applied ai\Assesment\utils\logs\yolo_human_filter.log"
os.makedirs(os.path.dirname(logfile), exist_ok=True)  # Ensure the directory exists

# Ensure the log file is created if it doesn't exist
if not os.path.exists(logfile):
    with open(logfile, 'w') as f:
        pass

# Configure logging
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
log = logging.getLogger()

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # Use "yolov8s.pt" or "yolov8m.pt" for better accuracy

def detect_human_yolo(image_path):
    try:
        results = model(image_path)  # Run inference

        for result in results:
            if result.boxes and len(result.boxes) > 0:  # Check if there are detections
                for box in result.boxes:
                    class_id = int(box.cls)  # Class index
                    if class_id == 0:  # Class 0 is "person" in COCO dataset
                        return True  # Human detected
        return False  # No human detected

    except Exception as e:
        logging.exception("Image is courrupted")
        return False  # Treat as non-human if error occurs
        
        
def filter_dataset(dataset_path, human_dir, non_human_dir):
    # Ensure output directories exist
    os.makedirs(human_dir, exist_ok=True)
    os.makedirs(non_human_dir, exist_ok=True)

    # Walk through the dataset directory
    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
                try:
                    if detect_human_yolo(file_path):
                        dest_path = os.path.join(human_dir, os.path.relpath(file_path, dataset_path))
                        logging.info(f"Human detected: {file}")
                    else:
                        dest_path = os.path.join(non_human_dir, os.path.relpath(file_path, dataset_path))
                        logging.info(f"No human detected: {file}")
                    
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    os.rename(file_path, dest_path)  # Move image

                except Exception as e:
                    logging.exception(f"Skipping {file} due to error: {e}")


# Example usage
dataset_path = r"D:\SHU\Applied ai\Assesment\dataset"
human_dir = r"D:\SHU\Applied ai\Assesment\filtered_data"
non_human_dir = r"D:\SHU\Applied ai\Assesment\non_filtered_data"

filter_dataset(dataset_path, human_dir,non_human_dir)
# image_path = "test.jpg"
# if detect_human_yolo(image_path):
#     print("Human detected!")
# else:
#     print("No human detected!")
