import numpy as np
import torch.nn as nn
from torch.utils import data
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import multilabel_confusion_matrix, classification_report
import matplotlib.pyplot as plt
from datetime import datetime
from config import *
from dataset_loader import CustomDataset, load_json_annotations
from model import CNN
from metrics import evaluate_metrics
import shutil
import inspect
import json
import seaborn as sns

# Main training loop
def train_model():
    timestamp = datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
    log_dir = os.path.join(BASE_DIR, "training_logs", f"{NICKNAME}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    script_dir = os.path.join(log_dir, "scripts")
    os.makedirs(script_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(log_dir, f"model_{NICKNAME}.pt")

    # Save full path to a text file
    path_txt_file = os.path.splitext(model_path)[0] + "_path.txt"  # same name, but "_path.txt"

    with open(path_txt_file, "w") as f:
        f.write(os.path.abspath(model_path))  # write full absolute path

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

    model = CNN(num_classes=len(label_map), image_size=IMAGE_SIZE).to(device)

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

    best_f1 = 0
    metrics_log = []  # Store metrics for each epoch
    total_loss = []
    ind = []
    f1_scores = []
    for epoch in range(n_epoch):

        metrics = evaluate_metrics(np.array(reals), np.array(preds))
        f1_scores.append(metrics['f1_macro'])
        epoch_metrics = {
            "epoch": epoch + 1,
            "loss": running_loss,
            "f1_macro": metrics["f1_macro"],
            "accuracy": metrics["accuracy"],
            "f1_micro": metrics.get("f1_micro"),
            "f1_weighted": metrics.get("f1_weighted"),
            "hamming_loss": metrics.get("hamming"),
            "cohen_kappa": metrics.get("cohen"),
            "matthews_corrcoef": metrics.get("mcc")
        }
        metrics_log.append(epoch_metrics)

        # Calculate multilabel confusion matrices
        conf_matrices = multilabel_confusion_matrix(np.array(reals), np.array(preds))

        # Create a directory to save confusion matrices if not exists
        conf_dir = os.path.join(log_dir, "confusion_matrices")
        os.makedirs(conf_dir, exist_ok=True)


        # Generate classification report
        report = classification_report(np.array(reals), np.array(preds), target_names=list(label_map.keys()),
                                       zero_division=0)

        # Save classification report to a text file
        with open(os.path.join(log_dir, "classification_report.txt"), "w") as f:
            f.write(report)

        print(
            f"Epoch {epoch + 1}: Loss={running_loss:.4f}, F1={metrics['f1_macro']:.4f}, Accuracy={metrics['accuracy']:.4f}")

        if metrics['f1_macro'] > best_f1 and SAVE_MODEL:
            best_f1 = metrics['f1_macro']
            torch.save(model.state_dict(), model_path)
            print("Model saved!")

            # Save each class confusion matrix separately
            for class_idx, cm in enumerate(conf_matrices):
                plt.figure(figsize=(4, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                            xticklabels=[f'Not {class_idx}', f'{class_idx}'],
                            yticklabels=[f'Not {class_idx}', f'{class_idx}'])
                plt.title(f'Confusion Matrix for Class {class_idx}')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.tight_layout()
                plt.savefig(os.path.join(conf_dir, f"confusion_matrix_class_{class_idx}_epoch_{epoch + 1}.png"))
                plt.close()

    with open(os.path.join(log_dir, "metrics_log.json"), "w") as f:
        json.dump(metrics_log, f, indent=4)
    print(f"Saved metrics log to {os.path.join(log_dir, 'metrics_log.json')}")

    plt.figure()
    plt.plot(total_loss)
    plt.title('Training Loss')
    plt.xlabel('Iterations')
    plt.savefig(os.path.join(log_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(f1_scores)
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.savefig(os.path.join(log_dir, "f1_score.png"))
    plt.close()

    # Access the three parallel conv layers
    conv_layers = {
        "7x7": model.conv1_7x7,
        "5x5": model.conv1_5x5,
        "3x3": model.conv1_3x3
    }

    # --- Visualize kernels ---
    for name, conv in conv_layers.items():
        weights = conv.weight.data.cpu().numpy()  # Shape: (out_channels, in_channels, k, k)
        num_filters = weights.shape[0]
        fig, axes = plt.subplots(nrows=num_filters // 4 + 1, ncols=4, figsize=(12, 3 * (num_filters // 4 + 1)))
        axes = axes.flatten()

        for i in range(num_filters):
            axes[i].imshow(weights[i, 0], cmap='gray')  # Show 1st channel of each filter
            axes[i].axis('off')
        for i in range(num_filters, len(axes)):
            axes[i].axis('off')  # Hide unused subplots

        plt.suptitle(f"Kernels from conv1_{name}")
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f"kernels_{name}.png"))
        plt.close()

    # --- Visualize feature maps on a sample image ---
    model.eval()
    with torch.no_grad():
        first_batch = next(iter(train_loader))
        image_tensor = first_batch[0][0].unsqueeze(0).to(device)  # Add batch dim

        for name, conv in conv_layers.items():
            feature_maps = conv(image_tensor).cpu().squeeze(0)  # Shape: (C, H, W)

            num_maps = feature_maps.shape[0]
            fig, axes = plt.subplots(nrows=num_maps // 4 + 1, ncols=4, figsize=(12, 3 * (num_maps // 4 + 1)))
            axes = axes.flatten()

            for i in range(num_maps):
                axes[i].imshow(feature_maps[i], cmap='gray')
                axes[i].axis('off')
            for i in range(num_maps, len(axes)):
                axes[i].axis('off')

            plt.suptitle(f"Feature maps from conv1_{name}")
            plt.tight_layout()
            plt.savefig(os.path.join(log_dir, f"feature_maps_{name}.png"))
            plt.close()


if __name__ == '__main__':
    train_model()
