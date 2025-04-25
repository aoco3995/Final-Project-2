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

class RandomZoomCropWithMetadata:
    def __init__(self, scale_range=(0.5, 1.0), image_size=224):
        self.scale_range = scale_range  # e.g., 0.5 to 1.0 means zoom from 50% to full image
        self.image_size = image_size

    def __call__(self, img: Image.Image):
        width, height = img.size

        # Random scale
        scale = random.uniform(*self.scale_range)
        crop_width = int(width * scale)
        crop_height = int(height * scale)

        # Random top-left coordinate (ensure it stays within bounds)
        max_left = width - crop_width
        max_top = height - crop_height
        left = random.randint(0, max_left) if max_left > 0 else 0
        top = random.randint(0, max_top) if max_top > 0 else 0
        right = left + crop_width
        bottom = top + crop_height

        # Crop and resize
        cropped_img = img.crop((left, top, right, bottom))
        resized_img = cropped_img.resize((self.image_size, self.image_size))

        # Metadata
        metadata = {
            "scale": scale,
            "crop_box": (left, top, right, bottom),
            "original_size": (width, height)
        }

        return resized_img, metadata



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
                        rows.append({"id": image, "target": label, "data": data[0]})
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
        change = random.randint(1,2)
        for ann in row['data']['annotations']:
            if (change == 1):
                x = random.randint(0, image.shape[1])
                y = random.randint(0, image.shape[0])
                row_target = row['target']
            else:
                x = int(math.floor(ann['coordinates']['x']))
                y = int(math.floor(ann['coordinates']['y']))
                row_target = 'none'
                

            w = int(ann['coordinates']['width'])
            h = int(ann['coordinates']['height'])
            image[(int(y-h/2)):(int(y+h/2)), (int(x-w/2)):(int(x+w/2)), :] = 0

        image = self.transform(image)
        target = [0] * len(self.label_map)
        for label in row_target.split(','):
            if label in self.label_map:
                target[self.label_map[label]] = 1
        return image, torch.FloatTensor(target)