import os
import torch
import numpy as np
from sklearn.metrics import classification_report
from config import *
from dataset_loader import CustomDataset, load_json_annotations
from model import CNN
from sklearn.preprocessing import MultiLabelBinarizer

# --- Load Model ---
model_path = "/home/ubuntu/Final-Project-2/Code/training_logs/MARS_2025_05_01_02:19:58/model_MARS.pt"
model = CNN(num_classes=4, image_size=IMAGE_SIZE).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- Load Test Data ---
df = load_json_annotations(JSON_FOLDER)

# Format targets
df = (
    df.groupby("id")
    .agg({
        "target": lambda x: ",".join(sorted(set(x))),
        "data":   "first"
    })
    .reset_index()
)

df['split'] = ['train' if i % 5 != 0 else 'test' for i in range(len(df))]

class_names = sorted(set(','.join(df['target']).split(',')))
label_map = {label: idx for idx, label in enumerate(class_names)}

test_df = df[df['split'] == 'test'].reset_index(drop=True)
test_set = CustomDataset(test_df, label_map)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# --- Generate Predictions ---
THRESHOLD = 0.5  # or use the one from config
preds, reals = [], []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds.extend((probs > THRESHOLD).astype(int))
        reals.extend(targets.numpy())

# --- Generate Classification Report ---
report = classification_report(np.array(reals), np.array(preds), target_names=list(label_map.keys()), zero_division=0)
print(report)

# Optional: save report
with open("classification_report.txt", "w") as f:
    f.write(report)
