import importlib
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import matplotlib.pyplot as plt
from datetime import datetime
from config import *
from dataset_loader import CustomDataset, load_json_annotations
from model import CNN
from metrics import evaluate_metrics
import shutil
import inspect

# Main training loop
def train_model():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(BASE_DIR, "training_logs", f"{NICKNAME}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    script_dir = os.path.join(log_dir, "scripts")
    os.makedirs(script_dir, exist_ok=True)

    source_files = [
        inspect.getfile(inspect.currentframe()),  # this script
        os.path.join(BASE_DIR, "config.py"),
        os.path.join(BASE_DIR, "dataset_loader.py"),
        os.path.join(BASE_DIR, "metrics.py"),
        os.path.join(BASE_DIR, "model.py")
    ]

    for src_path in source_files:
        if os.path.exists(src_path):
            dst_path = os.path.join(script_dir, os.path.basename(src_path).replace(".py", "_script.txt"))
            shutil.copy2(src_path, dst_path)
            print(f"Copied {src_path} to {dst_path}")
        else:
            print(f"Warning: {src_path} not found and was not copied.")

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

    df = (
        df
        .groupby("id")
        .agg({
            "target": lambda x: ",".join(sorted(set(x))),
            "data":   "first"
        })
        .reset_index()
    )

    #df = df.groupby('id')['target'].apply(lambda x: ','.join(sorted(set(x)))).reset_index()

    df['split'] = ['train' if i % 5 != 0 else 'test' for i in range(len(df))]

    mlb = MultiLabelBinarizer()
    class_names = sorted(set(','.join(df['target']).split(',')))
    class_names += ['none']
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
