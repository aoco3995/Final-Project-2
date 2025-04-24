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
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Configuration
IMAGE_SIZE = 100
BATCH_SIZE = 30
LR = 0.001
n_epoch = 10
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

# CNN Model
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4))
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

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

# Metrics

def evaluate_metrics(y_true, y_pred):
    return {
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'accuracy': accuracy_score(y_true, y_pred),
        'hamming': hamming_loss(y_true, y_pred),
        'cohen': cohen_kappa_score(y_true.argmax(axis=1), y_pred.argmax(axis=1)),
        'mcc': matthews_corrcoef(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    }

# Main training loop
def train_model():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(BASE_DIR, "training_logs", f"{NICKNAME}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Save hyperparameters
    hyperparams = {
        "IMAGE_SIZE": IMAGE_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": LR,
        "N_EPOCH": n_epoch,
        "THRESHOLD": THRESHOLD,
        "DEVICE": device
    }

    with open(os.path.join(log_dir, "hyperparameters.txt"), "w") as f:
        for key, val in hyperparams.items():
            f.write(f"{key}: {val}\n")

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

    # Length of Training Data
    print('The number of training examples is:', str(len(train_set)))
    # Length of Testing Data
    print('The number of testing examples is:', str(len(test_set)))
    # The shape of the first feature
    print('The shape of the first feature is:')
    print(train_set[0][0].shape)
    # The first label
    print('The first label is:' + str(train_set[0][1]))
    plt.figure()
    plt.imshow(train_set[0][0][0], cmap='gray')
    plt.savefig(os.path.join(log_dir, "first_feature.png"))
    plt.close()


    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = CNN(len(label_map)).to(device)
    print(model)
    # Save model architecture to txt
    model_info_path = os.path.join(log_dir, "model_architecture.txt")
    with open(model_info_path, "w") as f:
        f.write("Model Architecture:\n")
        f.write(str(model))
        f.write("\n\nTotal Parameters: {}\n".format(
            sum(p.numel() for p in model.parameters())
        ))
        f.write("Trainable Parameters: {}\n".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ))
    print(f"Saved model architecture to: {model_info_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0

    total_loss = []
    ind = []
    for epoch in range(n_epoch):
        model.train()
        running_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 100 == 0:
                total_loss.append(loss.item())
                ind.append(batch_idx + epoch * len(train_loader) / BATCH_SIZE)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(inputs), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

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
        print(f"Epoch {epoch+1}: Loss={running_loss:.4f}, F1={metrics['f1_macro']:.4f}, Accuracy={metrics['accuracy']:.4f}")

        if metrics['f1_macro'] > best_f1 and SAVE_MODEL:
            best_f1 = metrics['f1_macro']
            torch.save(model.state_dict(), f"model_{NICKNAME}.pt")
            print("Model saved!")

    plt.figure()
    plt.plot(total_loss)
    plt.title('Training Loss')
    plt.xlabel('Iterations')

    plt.savefig(os.path.join(log_dir, "loss_curve.png"))
    plt.close()

    # Access the first convolutional layer
    first_conv_layer = model.conv1

    # Get the weights of the layer
    weights = first_conv_layer.weight.data.cpu().numpy()

    # Visualize the kernels
    plt.figure()
    fig, axes = plt.subplots(nrows=8, ncols=4)
    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.imshow(weights[i, 0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "kernels.png"))
    plt.close()

    # Get the output of the first convolutional layer
    first_batch = next(iter(train_loader))
    image = first_batch[0][0].to(device)
    output = model.conv1(image).cpu()

    # Visualize feature maps
    plt.figure()
    for i in range(output.shape[0]):
        plt.subplot(8, 4, i + 1)  # Adjust grid size as needed
        plt.imshow(output[i].detach().numpy(), cmap='gray')
        plt.axis('off')
    plt.savefig(os.path.join(log_dir, "feature_maps.png"))
    plt.close()

if __name__ == '__main__':
    train_model()
