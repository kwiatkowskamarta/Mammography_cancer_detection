# Breast Cancer Detection in Mammograms using Deep Learning

## Project Overview
This project focuses on the automated detection and classification of breast tumors in mammographic scans using Convolutional Neural Networks (CNNs). Utilizing the MIAS (Mammographic Image Analysis Society) dataset, the system aims to assist radiologists by providing a "second opinion" on whether a detected lesion is Benign or Malignant.

The project implements a Transfer Learning approach using the ResNet50V2 architecture and compares two distinct experimental strategies:
1.  Global Classification: Classifying the entire mammogram (Normal vs. Benign vs. Malignant).
2.  ROI-Based Classification: Classifying specific Regions of Interest (Benign vs. Malignant) based on annotated coordinates.

Key Results: The ROI-based model achieved an Accuracy of 83% and a Malignant Recall (Sensitivity) of 87%, demonstrating strong potential for Computer-Aided Diagnosis (CAD) systems.

---

## Project Structure

```text
mammography-project/
│
├── data/
│   ├── raw/                 # Raw .pgm images and info.txt (Not included in repo)
│   ├── processed/           # Preprocessed full images
│   ├── processed_roi/       # Extracted ROI patches (Tumor crops)
│   ├── metadata_processed.csv
│   └── metadata_roi.csv
│
├── models/                  # Saved Keras models (.keras)
│
├── notebooks/
│   └── evaluation_results/  # Generated Plots (ROC, Confusion Matrix)
│
├── src/
│   ├── data_preprocessing.py  # Cleans metadata, resizes images, applies CLAHE
│   ├── create_roi_dataset.py  # Extracts Tumor ROIs based on ground truth
│   ├── train_model.py         # Exp 1: Trains Full-Image Model (3 classes)
│   ├── train_roi_model.py     # Exp 2: Trains ROI Model (Binary)
│   ├── evaluate_model.py      # Evaluation script for Exp 1
│   └── evaluate_roi_model.py  # Evaluation script for Exp 2
│
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
