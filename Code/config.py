import torch
import os

# Configuration
IMAGE_SIZE = 250
BATCH_SIZE = 30
LR = 0.0001
n_epoch = 1000
THRESHOLD = 0.5
SAVE_MODEL = True
NICKNAME = "MARS"

# Set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Get base path relative to current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Corrected Path to JSON annotations and images
JSON_FOLDER = os.path.join(BASE_DIR, "Dataset", "Aircraft_Fuselage_DET2023", "aircraft_fuselage_coco", "annotations")
DATA_DIR = os.path.join(BASE_DIR, "Dataset", "Aircraft_Fuselage_DET2023", "aircraft_fuselage_coco", "images")