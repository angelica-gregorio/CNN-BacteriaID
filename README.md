# CNN-BacteriaID

## Project Title
Edge Detection and Thresholding-Based Segmentation for CNN Classification of Gram-Stained Bacterial Species

## Objectives
- Preprocess bacterial microscopy images through RGB-to-HSV color space conversion.
- Segment bacterial cells from the background using Otsu's and adaptive thresholding, Canny edge detection, and morphological operations.
- Extract segmented bacterial regions as inputs for classification.
- Design and train a CNN for 33-class bacterial species classification.
- Apply data augmentation techniques to mitigate overfitting.
- Evaluate segmentation and classification performance using region isolation accuracy, accuracy, precision, recall, F1-score, and confusion matrices.

## Project Description
This project combines classical image processing with deep learning to classify Gram-stained bacterial microscopy images.

1. Convert RGB images to HSV to better separate Gram-positive and Gram-negative staining characteristics.
2. Segment bacteria from background using Otsu's and adaptive thresholding.
3. Refine boundaries and reduce noise using Canny edge detection and morphological operations.
4. Use connected component analysis to isolate bacterial regions and resize them into standardized patches.
5. Apply augmentation (rotation, flipping, brightness adjustment) to improve generalization and reduce overfitting.
6. Train a CNN to classify images into 33 bacterial species.
7. Evaluate performance with accuracy, precision, recall, F1-score, confusion matrices, and region isolation accuracy.
8. Compare classification performance between original images and segmented images to assess the impact of rule-based segmentation.

## Application of the Project
- Supports clinical microbiology laboratories with rapid preliminary bacterial identification from Gram-stained slides.
- Helps reduce diagnostic workload, especially in settings with limited specialist personnel.
- Enables potential use in portable point-of-care diagnostics for under-resourced healthcare environments.
- Contributes to antibiotic stewardship through faster species identification for timely treatment decisions.
- Serves as an educational demonstration of integrating thresholding and edge detection with CNN-based classification.