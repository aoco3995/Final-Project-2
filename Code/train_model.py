import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from sklearn.preprocessing import MultiLabelBinarizer
from model import CNN  # Import the CNN model from model.py
from config import *
from metrics import evaluate_metrics  # Import the metrics from metrics.py
from dataset_loader import load_json_annotations, CustomDataset  # Import the data loader functions


# Main training loop
def train_model():

    # Load JSON annotations and prepare the dataset
    df = load_json_annotations(JSON_FOLDER)

    # Group by 'id' and join targets, then create train/test split
    df = df.groupby('id')['target'].apply(lambda x: ','.join(sorted(set(x)))).reset_index()

    # Create a split for training and testing
    df['split'] = ['train' if i % 5 != 0 else 'test' for i in range(len(df))]

    mlb = MultiLabelBinarizer()

    # Convert targets to a binary matrix
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
