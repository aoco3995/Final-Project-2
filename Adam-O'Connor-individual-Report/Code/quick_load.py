from config import *
from dataset_loader import *
from train_model import *
from model import CNN
from metrics import evaluate_metrics
from matplotlib import pyplot as plt



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


df = load_json_annotations(JSON_FOLDER)

df_combined = (
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
label_map = {label: idx for idx, label in enumerate(class_names)}

train_df = df[df['split'] == 'train'].reset_index(drop=True)
test_df = df[df['split'] == 'test'].reset_index(drop=True)

train_set = CustomDataset(train_df, label_map)
test_set = CustomDataset(test_df, label_map)

def view_image(sample_index = 0):

    
    image_tensor, label_tensor, original_image, image_id = train_set[sample_index]

    # Convert label tensor to class names
    label_indices = (label_tensor == 1).nonzero(as_tuple=True)[0].tolist()
    label_names = [list(label_map.keys())[i] for i in label_indices]
    label_str = ', '.join(label_names) if label_names else 'None'

    # Show image
    plt.figure(figsize=(6, 6))
    plt.imshow(image_tensor.permute(1, 2, 0).numpy())
    plt.axis('off')

    # Overlay label and index
    plt.text(
        5, 20,
        f"Image #{image_id} - {label_str}",
        fontsize=12,
        color='white',
        bbox=dict(facecolor='black', alpha=0.7, boxstyle='round,pad=0.5')
    )

    # Save image
    plt.savefig('sample_image.png')
    plt.close()

    print(f"Saved labeled image as sample_image.png with label(s): {label_str}")
