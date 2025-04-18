from torch.utils import data
import os
import json
import cv2
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from config import *


# Load JSON annotations
def load_json_annotations(json_folder):
    rows = []
    for fname in os.listdir(json_folder):
        if fname.endswith(".json"):
            with open(os.path.join(json_folder, fname), "r") as f:
                data = json.load(f)
                for item in data:
                    image = item["image"]
                    for ann in item["annotations"]:
                        label = ann["label"]
                        rows.append({"id": image, "target": label})
    return pd.DataFrame(rows)

# Dataset Class
class CustomDataset(data.Dataset):
    def __init__(self, df, label_map):
        self.df = df
        self.label_map = label_map
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(DATA_DIR, row['id'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        target = [0] * len(self.label_map)
        for label in row['target'].split(','):
            if label in self.label_map:
                target[self.label_map[label]] = 1
        return image, torch.FloatTensor(target)
