![yogalense](https://github.com/user-attachments/assets/08a57bb0-a037-439c-80aa-e5e785e0089f)

# YogaLense🧘‍♀️
## AI-Powered Smart Yoga Coach

A computer vision system that combines deep learning and traditional machine learning to recognize yoga poses and provide corrective feedback. Built with CNN, YOLOv8, and MediaPipe.

---


## ⚡ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Extract features from dataset
python main.py

# 3. Get pose feedback (example)
python feedback_system.py -i samples/warrior_pose.jpg
```
Expected Outputs:

results/.../pose_features_with_labels.csv (feature dataset)

results/annotated/warrior_pose_annotated.jpg (extracted 33 keypoints)

*(Outputs appear in **results/feedback/logs/** folder)* 

## **📖 Table of Contents**
- Project Overview                                                                                                         
- System Architecture                                                                                                   
- Technical Details                                         

## Project Overview

This system performs two core functions:
1. **Pose Recognition:** Classifies input images into specific yoga poses using a Convolutional Neural Network (CNN).
2. **Form Feedback:** Analyzes body geometry using Logistic Regression with features extracted via MediaPipe to assess pose correctness.

---

## System Architecture

```plaintext
project-root/
├── datasets/
│   ├── non_filtered/                      # Unwanted yoga pose images
│   ├── processed/                         # Preprocessed training images
│   └── test/                              # testing images
├── models/                                # Serialized models
│   ├── yoga_pose_recognition.keras        # Trained CNN model
│   ├── correction_model_csv.pkl           # Logistic Regression Model
│   └──  yolov8n.pt                        # yolov8 Model            
├── src/
│   ├── preprocessor.py                    # Image preprocessing pipeline
│   ├── model.ipynb                        # Model training notebook
│   ├── feature_engineering.py             # Calculate angles and distance
│   ├── main.py                            # core algorithm logic
│   └── feedback_system.py                 # Feedback model logic
├── results/
│   ├── feedback/                          # Extracted pose features
│   ├── training/                          # Extracted CSV, annotated and logs : training
│   └── test/                              # Extracted CSV, annotated and logs : test
├── utils/
│   ├── image_downloader.py                # Script to download images
│   ├── renames_images.py                  # Script to rename images
│   ├── yolo_human_filter.py               # YOLO-based human filtering
└── requirements.txt                       # Python dependencies
```

---

## Key Features

- **Multi-Stage Processing Pipeline**
  - Image resizing and normalization
  - Keypoint extraction using MediaPipe Pose
  - feature engineering using formulas (joint angles, limb distances)
- **Dual-Model Architecture**
  - CNN for visual pattern recognition(Categorial)
  - Logistic Regression for labeling (Correct/Incorrect)
- **Detailed Correctness Probability Scores**:
  - The system provides a probability score indicating how likely the pose is "correct" or "incorrect."
  - Example: "Your Cobra Pose is 85% correct."
- **Error Localization in Pose Execution**:
  - Identifies specific areas of the pose that need improvement.
  - Example: "Adjust your left elbow angle to 90 degrees."

---

## Technical Requirements

- Python 3.8+
- TensorFlow 2.7+
- MediaPipe 0.8.9+
- scikit-learn 1.0+
- OpenCV 4.5+

```bash
# Install dependencies
pip install -r requirements.txt
```

---

## Implementation Workflow

### 1. Data Collection
```bash
python utils/image_downloader.py \
 --output_dir data/dataset\

python utils/ renames_images.py \
 --input_dir datasets/dataset \ 
 --output_dir datasets/dataset \

python utils/yolo_human_flter.py \
 --input_dir datasets/dataset \
 --output_dir datasets/filtered \

```
- Download raw dataset which is provided by author
- Filters Human Poses: Uses YOLO-based human detection to filter out irrelevant images.

### 1. Data Preparation
```bash
python src/preprocessor.py \
  --input_dir datasets/filtered \
  --output_dir datasets/processed \
  
```
- he script resizes images to 224x224, normalizes pixel values to [0, 1]
- Uses MediaPipe Pose to extract body landmarks, calculates mean visibility, and saves annotated images for visualization and debugging.

### 2. Model Training
```bash
jupyter notebook src/model.ipynb
```
- CNN trained with Adam optimizer
- LR model uses L2 regularization

### 3. System Execution
```bash
python src/feedback_system.py \
  --input test_data/tree_pose.jpg \   --- model: CNN (pose name)
 --input test_data/CSV_sheet \        --- model: logistic regression(correctness feedback)
  --output_dir results/
```
- Generates log file with images name: Predication pose.
- Classification: Correct/Incorrect with correctness score
- (for eg. Predicted pose-- bridge_pose Predicted correctness: correct (74.88%))
- Updates system logs in `feedback/logs/`

---

## Mathematical Foundations

**Joint Angle Calculation**  
For three body landmarks \( A(x_1,y_1) \), \( B(x_2,y_2) \), \( C(x_3,y_3) \):

\[
\theta = \arccos\left(\frac{BA \cdot BC}{\|BA\|\|BC\|}\right)
\]

\[
BA = (x_1-x_2, y_1-y_2), \quad BC = (x_3-x_2, y_3-y_2)
\]

**Distance Calculation**                                                                                
For two keypoints ( A(x_1, y_1) ) and ( B(x_2, y_2) ):      
                                         
[ \text{distance} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} ]

**Normalization of Keypoints**                                                                                                
For a keypoint ( (x, y) ) in an image of dimensions (height & width)                                                                        

[ x_{\text{normalized}} = \frac{x}{\text{image_width}}, \quad y_{\text{normalized}}                         
                                     =\frac{y}{\text{image_height}} ]

**Pose Labeling (Range Checking)**
For a computed feature value ( v ) (e.g., angle or distance), the pose is labeled as "correct" if:    
[ \text{min_val} \leq v \leq \text{max_val} ]

Otherwise, the pose is labeled as "incorrect."


---

## Performance Metrics
![mean_visibility_distribution](https://github.com/user-attachments/assets/800cad9e-ffdd-42a1-93dc-c6b7dcb1ae03)
### CNN Model:
![model](https://github.com/user-attachments/assets/ee3794ed-d333-45d0-9ae8-4aafb72edae2)
![matrix](https://github.com/user-attachments/assets/5935f724-530e-4f3e-9249-fa3f1b34713f)
### Logistic Regression Model:
![output](https://github.com/user-attachments/assets/3e190278-4490-4223-bb41-c72062aa7afd)
![accuracy](https://github.com/user-attachments/assets/5992e527-5de9-4cf0-95b6-3386f6ebac16)








## Roadmap & Future Development

- [ ] Real-time video analysis module
- [ ] Mobile-optimized inference (TensorFlow Lite)
- [ ] 3D pose estimation integration
- [ ] Personalized adaptation engine

---
Enjoy exploring the project!🥂
