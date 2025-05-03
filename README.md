# Final Project X: Aircraft Fuselage Defect Detection Using CNN

## Repository Structure

```
Final-Project-X/
├── Proposal/
│   └── Final_Proposal.pdf
├── Final-Group-Project-Report/
│   └── Final_Group_Report.pdf
├── Final-Presentation/
│   └── Final_Presentation.pdf
├── Code/
│   ├── config.py
│   ├── dataSet_count.py
│   ├── dataset_loader.py
│   ├── get_class_report.py
│   ├── get_image.py
│   ├── load_model.py
│   ├── metrics.py
│   ├── model.py
│   ├── quick_load.py
│   ├── train_model.py
│   ├── Unlabeled_test_MARS.py
│   ├── View_images.py
│   ├── pseudo_labels_MARS.csv
│   ├── labeled.jpg
│   ├── sample_image.png
│   ├── __pycache__/
│   ├── Saved_logs/
│   └── training_logs/
├── Dataset/
│   └── Aircraft_Fuselage_DET2023/
│       ├── aircraft_fuselage_coco/
│       │   ├── annotations/
│       │   └── images/
│       ├── aircraft_fuselage_voc/
│       ├── aircraft_fuselage_yolo/
│       └── unlabel_aircraft_fuselage/
├── Adam-O'Connor-individual-Report/
│   └── Individual_Report.pdf
├── README.md
```

## Overview

This repository contains the materials for Final Project: Aircraft Fuselage Defect Detection Using CNN, completed for ECEN 5060 Deep Learning, Dr. Hagan. The project focuses on detecting defects in aircraft fuselage images using convolutional neural networks (CNNs). The goal is to automate the identification of structural defects to improve inspection efficiency and safety.

## Team Members

- Adam O'Connor
- Jake Whited

## Code Instructions

To train the model, first set your desired parameters in `config.py` (e.g., learning rate, batch size, number of epochs, and dataset paths). Once configured, simply run `train_model.py` to begin the training process. During training, a timestamped folder will be created under `Saved_logs/training_logs/`, where all logs, visualizations, and a copy of the code used will be stored along with the best-performing model. Logs and saved models will be stored automatically.

All code related to the final project is located in the top-level `Code/` directory. Below is a list of the key Python scripts and their purposes:

- `config.py` -
  Stores configuration parameters such as:
  - `IMAGE_SIZE`, `BATCH_SIZE`, `LR`, `n_epoch`, `THRESHOLD`
  - Device setup (`cuda` or `cpu` depending on availability)
  - Dataset paths to COCO JSON annotations and image files
  This file ensures centralized control over training settings and dataset location.
- `dataSet_count.py` - Analyzes and counts dataset elements.
- `dataset_loader.py` - Loads and prepares the dataset.
- `get_class_report.py` - Generates classification reports.
- `get_image.py` - Utility script for image retrieval.
- `load_model.py` - Loads the trained CNN model.
- `metrics.py` - Contains performance metric calculations.
- `model.py` - Defines the CNN architecture.
- `quick_load.py` - Quickly loads data and models for testing.
- `train_model.py` - Trains the CNN using the labeled dataset.
- `Unlabeled_test_MARS.py` - 
  After a model has been successfully trained, this script is used to evaluate the model's performance on an unlabeled dataset located in `unlabel_aircraft_fuselage/`. It runs inference on these images and generates pseudo labels, which are saved to a file named `pseudo_labels_MARS.csv`. This allows for further analysis or potential semi-supervised learning strategies using the newly labeled data.
- `View_images.py` -
  Used to visualize the pseudo-labeled results from the model. Set the `TARGET_LABEL` variable (line 10) to the defect type you want to examine (e.g., `'scratch'`). The script will search the `pseudo_labels_MARS.csv` file for matching entries and display the corresponding images from the unlabeled dataset. It saves each labeled image to `labeled.jpg` with the predicted label overlaid. This helps in manually verifying how well the model is labeling unseen data.

Other supporting files:
- `pseudo_labels_MARS.csv` - CSV file with model-generated labels.
- `labeled.jpg` and `sample_image.png` - Sample images used for testing or visualization.

Logs:
- `Saved_logs/` - Contains subdirectories of training logs for each run.
  - Each run is timestamped and includes all output artifacts from that specific training session.
  - Contents include:
    - Confusion matrices for each class (`confusion_matrix_class_X_epoch_Y.png`)
    - F1 score and loss curve visualizations
    - Feature maps and kernel visualizations
    - Model architecture (`model_MARS.pt`, `model_MARS_path.txt`)
    - Full training logs and metric summaries (`metrics_log.json`, `classification_report.txt`)
    - Copies of all scripts used during the run (`scripts/` folder) for reproducibility
  - This system ensures each model run is fully documented and reproducible.
- `training_logs/` - Directory for training output logs.

## Dataset

The dataset used in this project is stored in `Dataset/Aircraft_Fuselage_DET2023/` and includes data in multiple annotation formats:

The primary dataset used for training and evaluation is the COCO-formatted dataset located at `aircraft_fuselage_coco/`. This directory contains:
- `annotations/` — JSON files containing COCO-style object detection annotations.
- `images/` — Corresponding image files of aircraft fuselages.

The project pipeline is designed to parse COCO annotations to extract bounding boxes and labels, which are then used to train the CNN model. The `dataset_loader.py` script handles this parsing and transforms the images and labels into PyTorch-compatible datasets, applying necessary augmentations and preprocessing.

- `aircraft_fuselage_coco/` - COCO-style annotations and images
- `aircraft_fuselage_voc/` - Pascal VOC format
- `aircraft_fuselage_yolo/` - YOLO format
- `unlabel_aircraft_fuselage/` - Unlabeled images for testing or augmentation

## Proposal

The project proposal is located in the `Proposal/` directory as `Final_Proposal.pdf`.

## Final Report

The group's final report is located in the `Final-Group-Project-Report/` directory.

## Final Presentation

The project presentation is saved as a PDF in the `Final-Presentation/` folder.


