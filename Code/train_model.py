import os
import json
import random
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision import transforms
from tqdm import tqdm
from model import CNN  # Import the CNN model from model.py
from config import *
from metrics import evaluate_metrics  # Import the metrics from metrics.py


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


# Main training loop
def train_model():
    df = load_json_annotations(JSON_FOLDER)
    df = df.groupby('id')['target'].apply(lambda x: ','.join(sorted(set(x)))).reset_index()
    df['split'] = ['train' if i % 5 != 0 else 'test' for i in range(len(df))]

    mlb = MultiLabelBinarizer()
    class_names = sorted(set(','.join(df['target']).split(',')))
    label_map = {label: idx for idx, label in enumerate(class_names)}

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)

    train_set = CustomDataset(train_df, label_map)
    test_set = CustomDataset(test_df, label_map)

    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN(len(label_map)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0

    for epoch in range(n_epoch):
        model.train()
        running_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        preds, reals = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds.extend((probs > THRESHOLD).astype(int))
                reals.extend(targets.numpy())

        metrics = evaluate_metrics(np.array(reals), np.array(preds))
        print(f"Epoch {epoch+1}: Loss={running_loss:.4f}, F1={metrics['f1_macro']:.4f}")

        if metrics['f1_macro'] > best_f1 and SAVE_MODEL:
            best_f1 = metrics['f1_macro']
            torch.save(model.state_dict(), f"model_{NICKNAME}.pt")
            print("Model saved!")

if __name__ == '__main__':
    train_model()
