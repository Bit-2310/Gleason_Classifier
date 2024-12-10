# Automated Gleason Scoring System

This repository contains an implementation of an automated Gleason scoring system using convolutional neural network (CNN) models such as **ResNet-50** and **EfficientNet**. The aim is to classify prostate tissue into Gleason scores to assist in the diagnosis of prostate cancer.

## Overview

### Models Used
- **ResNet-50**: A pretrained CNN model used for image classification. It achieved an accuracy of 97% during testing, demonstrating high reliability in predicting Gleason grades.
- **EfficientNet**: Another CNN model used for comparison. It achieved an accuracy of 62%, showing moderate performance.

### Visualization
- **Grad-CAM**: Grad-CAM visualizations were used to highlight areas of tissue predicted to have tumors. This helps understand what features the model focuses on while making predictions.

## Dataset
- The dataset used contains **200 tissue microarray (TMA) images**, split in a **70:15:15** ratio for training, validation, and testing.
- The images were labeled into **4 categories**: Benign, Gleason3, Gleason4, and Gleason5.

## Methodology

1. **Data Splitting and Preprocessing**
   - The dataset was split into training, validation, and test sets.
   - Images were preprocessed by resizing, normalization, and augmentation.

2. **Model Training and Testing**
   - **ResNet-50** and **EfficientNet** were trained for **10 epochs** each.
   - **Dropout** and **batch normalization** were used to prevent overfitting.
   - **Learning rate scheduler** was applied to optimize the learning process.

3. **Visualization**
   - **Grad-CAM** was applied to visualize model predictions on various tissue samples.
   - Example visualizations and misclassified images are available to better understand the model behavior.

## Results
- **ResNet-50** achieved a test accuracy of **97%**, showing high reliability.
- **EfficientNet** achieved a test accuracy of **62%**, with room for further optimization.
- Grad-CAM visualization helped in identifying the regions that influenced the predictions the most.

## Future Improvements
- Increase dataset size to enhance model robustness.
- Experiment with other architectures like **MobileNet** or **DenseNet** to identify the most suitable model for resource-constrained environments.
- Create **ensemble models** to improve overall prediction accuracy.

## Repository Structure
- **/data**: Contains the TMA dataset (not included due to size restrictions).
- **/src**: Contains all Python scripts for model training, testing, and visualization.
- **README.md**: Project description and instructions for use.

## Usage Instructions
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/gleason-scoring-system.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the main script for training and testing:
   ```
   python src/main.py
   ```

## License
This project is licensed under the MIT License.

## Acknowledgments
- The models are based on the PyTorch framework and use pretrained weights from **torchvision**.
- Grad-CAM implementation was inspired by related research in model interpretability.

## Contact
For any questions or collaborations, please reach out at your.email@example.com.
