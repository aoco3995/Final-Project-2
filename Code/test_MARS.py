import os
import cv2
import torch
import torch.nn as nn
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

IMAGE_SIZE = 100
NUM_CLASSES = 4
NICKNAME = "MARS"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

UNLABELED_IMAGES = [...]  # list of 5000 image file names
LABEL_MAP = {...}  # Your label -> index dictionary
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Dataset", "Aircraft_Fuselage_DET2023", "unlabel_aircraft_fuselage")

n_epoch = 1
BATCH_SIZE = 30
LR = 0.001

mlb = MultiLabelBinarizer()
THRESHOLD = 0.5
SAVE_MODEL = True

#------------------------------------------------------------------------------------------------------------------
#---- Define the model ---- #

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pad1 = nn.ZeroPad2d(2)

        self.conv2 = nn.Conv2d(16, 128, (3, 3))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(128, NUM_CLASSES)
        self.act = torch.relu

    def forward(self, x):
        x = self.pad1(self.convnorm1(self.act(self.conv1(x))))
        x = self.act(self.conv2(self.act(x)))
        return self.linear(self.global_avg_pool(x).view(-1, 128))


#------------------------------------------------------------------------------------------------------------------

class ModelTester:
    def __init__(self, model, model_path, label_map, image_dir, image_size=100, threshold=0.5, device=None):
        self.model = model
        self.model_path = model_path
        self.label_map = label_map
        self.rev_label_map = {v: k for k, v in label_map.items()}
        self.image_dir = image_dir
        self.image_size = image_size
        self.threshold = threshold
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        self._load_model()

    def _load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict_image(self, img_path):
        img = cv2.imread(os.path.join(self.image_dir, img_path))
        if img is None:
            return []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = torch.sigmoid(self.model(img_tensor)).cpu().numpy()[0]

        predicted_labels = [
            self.rev_label_map[i] for i, prob in enumerate(output) if prob > self.threshold
        ]
        return predicted_labels

    def run_on_images(self, image_list, save_csv=None, confidence_threshold=0.9):
        results = []
        for img_name in tqdm(image_list, desc="Generating pseudo-labels"):
            preds = self.predict_image(img_name)
            if preds:
                results.append({"id": img_name, "target": ",".join(preds)})

        df_results = pd.DataFrame(results)
        if save_csv:
            df_results.to_csv(save_csv, index=False)
        return df_results

    # ------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    model = CNN()
    MODEL_PATH = model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))
    tester = ModelTester(model, MODEL_PATH, LABEL_MAP, DATA_DIR, image_size=IMAGE_SIZE)

    pseudo_labels_df = tester.run_on_images(UNLABELED_IMAGES, save_csv="pseudo_labels.csv")
