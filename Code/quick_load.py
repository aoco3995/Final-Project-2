from config import *
from dataset_loader import *
from train_model import *
from model import CNN
from metrics import evaluate_metrics



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