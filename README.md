# 🦠 BacteriaID-CNN

**Edge Detection and Thresholding-Based Segmentation for CNN Classification of Gram-Stained Bacterial Species**

A research project that combines **classical image processing** and **deep learning** to automatically identify bacterial species from Gram-stained microscopy images. The proposed pipeline enhances bacterial cell segmentation using edge detection and thresholding before performing classification with a Convolutional Neural Network (CNN).

---

## 📖 Overview

Microscopic bacterial identification is a fundamental task in clinical microbiology but is often labor-intensive and dependent on experienced laboratory personnel. This project investigates whether incorporating rule-based image segmentation techniques prior to deep learning classification can improve the performance of bacterial species recognition.

The pipeline first preprocesses Gram-stained microscopy images through color space conversion and image segmentation. The segmented bacterial regions are then used to train a CNN capable of classifying **33 bacterial species**.

---

## 🎯 Objectives

- Convert bacterial microscopy images from **RGB to HSV** color space.
- Segment bacterial cells using:
  - Otsu's Thresholding
  - Adaptive Thresholding
  - Canny Edge Detection
  - Morphological Operations
- Extract bacterial regions using connected component analysis.
- Train a CNN to classify **33 Gram-stained bacterial species**.
- Apply data augmentation techniques to improve model generalization.
- Compare CNN performance using:
  - Original microscopy images
  - Segmented bacterial images
- Evaluate the system using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
  - Region Isolation Accuracy

---

## 🔬 Methodology

The proposed workflow consists of the following stages:

```text
Gram-Stained Microscopy Images
                │
                ▼
       RGB → HSV Conversion
                │
                ▼
     Image Thresholding
 (Otsu + Adaptive Thresholding)
                │
                ▼
      Canny Edge Detection
                │
                ▼
    Morphological Operations
                │
                ▼
 Connected Component Analysis
                │
                ▼
 Region Extraction & Resizing
                │
                ▼
      Data Augmentation
                │
                ▼
     CNN Classification
                │
                ▼
   33 Bacterial Species
```

---

## 🧠 Technologies

### Image Processing

- OpenCV
- NumPy
- RGB-to-HSV Conversion
- Otsu Thresholding
- Adaptive Thresholding
- Canny Edge Detection
- Morphological Operations
- Connected Component Analysis

### Deep Learning

- TensorFlow / Keras
- Convolutional Neural Networks (CNN)
- Data Augmentation

### Programming Language

- Python

---

## 📂 Project Structure

```text
BacteriaID-CNN/
│
├── dataset/
│   ├── raw/
│   ├── segmented/
│   └── augmented/
│
├── preprocessing/
│   ├── rgb_to_hsv.py
│   ├── thresholding.py
│   ├── edge_detection.py
│   ├── morphology.py
│   └── segmentation.py
│
├── models/
│   ├── cnn_model.py
│   └── train.py
│
├── evaluation/
│   ├── metrics.py
│   ├── confusion_matrix.py
│   └── visualization.py
│
├── notebooks/
│
├── results/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📊 Evaluation Metrics

The proposed system will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Region Isolation Accuracy

The study also compares the classification performance between:

- Original Gram-stained microscopy images
- Segmented bacterial cell images

to determine whether classical image segmentation improves CNN classification.

---

## 💡 Applications

This project has potential applications in:

- 🏥 Clinical microbiology laboratories
- ⚕️ Computer-aided bacterial identification
- 🌍 Point-of-care diagnostic systems
- 💊 Antibiotic stewardship
- 🎓 Microbiology education
- 🤖 Medical image analysis research

---

## 👥 Authors

- Christian John B. Alado
- Angelica G. Gregorio
- Andrei Gyles S. Lim
- Jared Joshua A. Lofamia
- Allan John C. Olivete

**De La Salle University – Manila**  
Gokongwei College of Engineering

---

## 📄 License

This repository is intended for academic and research purposes.
