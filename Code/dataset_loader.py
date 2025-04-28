from torch.utils import data
import os
import json
import cv2
import pandas as pd
import torch
from torchvision import transforms
from config import *
from PIL import Image
import random
import torch
import math
import random
import numpy as np

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
        rows.append({"id": image, "target": label, "data": data[0]})  # Corrected: inside the loop

    # Sort rows by "id"
    rows = sorted(rows, key=lambda x: x['id'])

    # Count current samples per class
    class_counts = {
        'rust': 0,
        'scratch': 0,
        'rivet_damage': 0,
        'paint_peel': 0
    }

    for row in rows:
        if row['target'] in class_counts:
            class_counts[row['target']] += 1

    print(f"Current class counts: {class_counts}")

    # Find the maximum class size
    max_count = max(class_counts.values())

    # Oversample: duplicate rows for underrepresented classes
    balanced_rows = []

    for target_class in class_counts.keys():
        # Extract all rows with this target
        class_rows = [row for row in rows if row['target'] == target_class]
        
        # Duplicate as needed
        if len(class_rows) < max_count:
            multiplier = max_count // len(class_rows)
            remainder = max_count % len(class_rows)
            balanced_class_rows = class_rows * multiplier + random.sample(class_rows, remainder)
        else:
            balanced_class_rows = class_rows
        
        balanced_rows.extend(balanced_class_rows)

    print(f"Balanced dataset size: {len(balanced_rows)}")

    rows = balanced_rows

    class_counts = {
        'rust': 0,
        'scratch': 0,
        'rivet_damage': 0,
        'paint_peel': 0
    }

    for row in rows:
        if row['target'] in class_counts:
            class_counts[row['target']] += 1

    print(f"Current class counts: {class_counts}")

    # Shuffle for randomness
    random.shuffle(balanced_rows)

    return pd.DataFrame(balanced_rows)

# Dataset Class
class CustomDataset(data.Dataset):
    def __init__(self, df, label_map):
        self.df = df
        self.label_map = label_map
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),           # Randomly flip horizontally
            transforms.RandomVerticalFlip(p=0.2),             # Randomly flip vertically (less common)
            transforms.RandomResizedCrop(
                size=(IMAGE_SIZE, IMAGE_SIZE),
                scale=(0.8, 1.0),   # Random zoom between 80% and 100%
                ratio=(0.9, 1.1)    # Allow a little squishing/stretching
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.02           # Slight color variations
            ),
            transforms.RandomRotation(degrees=10),             # Small random rotations
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(DATA_DIR, row['id'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Save a copy of the *original* image to pull from
        original_image = image.copy()

        change = 2

        black_image = np.zeros_like(image)

        collected_targets = set()

        for ann in row['data']['annotations']:
            if change > 1:
                x = int(math.floor(ann['coordinates']['x']))
                y = int(math.floor(ann['coordinates']['y']))
                current_target = row['target']
            else:
                x = random.randint(0, image.shape[1])
                y = random.randint(0, image.shape[0])
                current_target = 'none'

            w = int(ann['coordinates']['width'])
            h = int(ann['coordinates']['height'])

            y1 = max(int(y - h / 2), 0)
            y2 = min(int(y + h / 2), image.shape[0])
            x1 = max(int(x - w / 2), 0)
            x2 = min(int(x + w / 2), image.shape[1])

            # Important: Always copy from the original unmodified image
            black_image[y1:y2, x1:x2, :] = original_image[y1:y2, x1:x2, :]

            for label in current_target.split(','):
                collected_targets.add(label)

        image = black_image

        image = self.transform(image)

        target = [0] * len(self.label_map)
        for label in collected_targets:
            if label in self.label_map:
                target[self.label_map[label]] = 1

        return image, torch.FloatTensor(target)