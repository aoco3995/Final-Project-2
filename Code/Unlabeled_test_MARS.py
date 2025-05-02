import os
import cv2
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
import random
import math
import torch.nn.functional as F

IMAGE_SIZE = 250
NUM_CLASSES = 4
NICKNAME = "MARS"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

LABEL_MAP = {
    "Scratch": 0,
    "Paint Peel": 1,
    "Rivet Damage": 2,
    "Rust": 3,
    "none": 4
}  # Your label -> index dictionary
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Dataset", "Aircraft_Fuselage_DET2023", "unlabel_aircraft_fuselage")

UNLABELED_IMAGES = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

n_epoch = 1
BATCH_SIZE = 30
LR = 0.001

mlb = MultiLabelBinarizer()
THRESHOLD = 0.4
SAVE_MODEL = True

#------------------------------------------------------------------------------------------------------------------
#---- Define the model ---- #

class CNN(nn.Module):
    def __init__(self, num_classes, image_size):
        super(CNN, self).__init__()

        # === PARALLEL CONVS (input: 3 channels) ===
        self.conv1_7x7 = nn.Conv2d(3, 8, kernel_size=7, stride=1, padding=3)
        self.conv1_5x5 = nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2)
        self.conv1_3x3 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        # Total output after concat: 8+8+8 = 24 channels

        # === CONV + POOL ===
        self.conv2 = nn.Conv2d(24, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # === DYNAMIC FC CALCULATION ===
        dummy_input = torch.zeros(1, 3, image_size, image_size)
        x = self._forward_conv_layers(dummy_input)
        flattened_size = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _forward_conv_layers(self, x):
        # Apply all 3 in parallel
        out_7x7 = F.relu(self.conv1_7x7(x))
        out_5x5 = F.relu(self.conv1_5x5(x))
        out_3x3 = F.relu(self.conv1_3x3(x))

        # Concatenate along the channel dimension
        x = torch.cat((out_7x7, out_5x5, out_3x3), dim=1)  # Shape: (B, 24, H, W)

        # Continue through rest of CNN
        x = self.pool(F.relu(self.conv2(x)))  # Downsample
        x = self.pool(x)  # Downsample again
        return x

    def forward(self, x):
        x = self._forward_conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Dataset Class
class UnlabeledDataset(data.Dataset):
    def __init__(self, image_list, DATA_DIR):
        self.image_list = image_list
        self.DATA_DIR = DATA_DIR
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
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.DATA_DIR, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image)

        return image, img_name



class ModelTester:
    def __init__(self, model, model_path, label_map, threshold=0.5, image_size=100, batch_size=32, device=None):
        self.model = model
        self.model_path = model_path
        self.label_map = LABEL_MAP
        self.rev_label_map = {v: k for k, v in label_map.items()}
        self.threshold = THRESHOLD
        self.image_size = IMAGE_SIZE
        self.batch_size = BATCH_SIZE
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()

    def _load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def run(self, image_list, image_dir, save_csv=None):
        dataset = UnlabeledDataset(image_list, image_dir)
        dataloader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        results = []

        with torch.no_grad():
            for images, img_names in tqdm(dataloader, desc="Running Inference"):
                images = images.to(self.device)
                outputs = torch.sigmoid(self.model(images)).cpu().numpy()

                for i, output in enumerate(outputs):
                    preds = [self.rev_label_map[j] for j, prob in enumerate(output) if prob > self.threshold]
                    if not preds:  # fallback: use the most confident class
                        max_index = np.argmax(output)
                        preds = [self.rev_label_map[max_index]]
                    results.append({"id": img_names[i], "target": ",".join(preds)})

        df_results = pd.DataFrame(results)
        if save_csv:
            df_results.to_csv(save_csv, index=False)
        return df_results

def manual_grading(df, num_samples=10, save_corrected_csv='corrected_labels.csv'):
    correct = 0
    results = []

    sampled = df.sample(n=num_samples).reset_index(drop=True)

    for i, row in sampled.iterrows():
        img_path = os.path.join(DATA_DIR, row['id'])
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Show the image and predicted label
        plt.imshow(image_rgb)
        plt.title(f"[{i + 1}/{num_samples}] Predicted: {row['target']}")
        plt.axis('off')
        plt.show()

        # Prompt user for input
        user_input = input(
            "Enter correct label(s) (comma-separated), or press Enter to accept prediction: ").strip()

        if user_input:
            user_labels = ",".join([label.strip() for label in user_input.split(",")])
        else:
            user_labels = row['target']  # Accept prediction

        predicted_set = set(row['target'].split(","))
        actual_set = set(user_labels.split(","))

        is_correct = predicted_set == actual_set
        if is_correct:
            correct += 1
        else:
            print(f"❌ Mismatch → Model: {predicted_set} | You: {actual_set}")

        results.append({"id": row['id'], "predicted": row['target'], "corrected": user_labels})

    # Save corrected results
    df_corrected = pd.DataFrame(results)
    df_corrected.to_csv(save_corrected_csv, index=False)

    print(f"\n✅ Manual grading complete. Accuracy: {correct}/{num_samples} = {correct / num_samples * 100:.2f}%")
    print(f"📝 Corrected labels saved to: {save_corrected_csv}")

    # ------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    model = CNN(num_classes=NUM_CLASSES, image_size=IMAGE_SIZE)
    tester = ModelTester(model=model,
                         model_path='/home/ubuntu/Final-Project-2/Code/training_logs/MARS_2025_05_01_02:19:58/model_MARS.pt',
                         label_map=LABEL_MAP,
                         threshold=THRESHOLD,
                         image_size=IMAGE_SIZE,
                         batch_size=BATCH_SIZE,
                         device=device)

    df_pseudo_labels = tester.run(
        image_list=UNLABELED_IMAGES,
        image_dir=DATA_DIR,
        save_csv='pseudo_labels_{}.csv'.format(NICKNAME)
    )

    print("✅ Saved pseudo-labels for {} images".format(len(df_pseudo_labels)))

    manual_grading(df_pseudo_labels, num_samples=10, save_corrected_csv='corrected_labels_MARS.csv')