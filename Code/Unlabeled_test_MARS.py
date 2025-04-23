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

IMAGE_SIZE = 100
NUM_CLASSES = 4
NICKNAME = "MARS"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

LABEL_MAP = {
    "Crack": 0,
    "Dent": 1,
    "Paint_Peel": 2,
    "Scratch": 3
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
THRESHOLD = 0.5
SAVE_MODEL = True

#------------------------------------------------------------------------------------------------------------------
#---- Define the model ---- #

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
class UnlabeledDataset(data.Dataset):
    def __init__(self, image_list, DATA_DIR, image_size=100):
        self.image_list = image_list
        self.DATA_DIR = DATA_DIR
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
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
        self.label_map = label_map
        self.rev_label_map = {v: k for k, v in label_map.items()}
        self.threshold = threshold
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()

    def _load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def run(self, image_list, image_dir, save_csv=None):
        dataset = UnlabeledDataset(image_list, image_dir, image_size=self.image_size)
        dataloader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        results = []

        with torch.no_grad():
            for images, img_names in tqdm(dataloader, desc="Running Inference"):
                images = images.to(self.device)
                outputs = torch.sigmoid(self.model(images)).cpu().numpy()

                for i, output in enumerate(outputs):
                    preds = [self.rev_label_map[j] for j, prob in enumerate(output) if prob > self.threshold]
                    results.append({"id": img_names[i], "target": ",".join(preds)})

        df_results = pd.DataFrame(results)
        if save_csv:
            df_results.to_csv(save_csv, index=False)
        return df_results

    # ------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    model = CNN(num_classes=NUM_CLASSES)
    tester = ModelTester(model=model,
                         model_path='model_{}.pt'.format(NICKNAME),
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
