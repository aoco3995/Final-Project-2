import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

n_epoch = 2
BATCH_SIZE = 32
LR = 0.001

## Image processing
CHANNELS = 3
IMAGE_WIDTH = 1120
IMAGE_HEIGHT = 960

class AircraftFuselageDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, label_list=None, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform

        # Match all image filenames (assuming .jpg)
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])

        # Build label vocabulary (optional: if label_list is passed, use it)
        self.label_set = set()
        if label_list is None:
            for img_file in self.image_files:
                json_file = os.path.splitext(img_file)[0] + '.json'
                with open(os.path.join(self.annotation_dir, json_file)) as f:
                    data = json.load(f)[0]
                    for ann in data['annotations']:
                        self.label_set.add(ann['label'])
            self.labels = sorted(list(self.label_set))
        else:
            self.labels = sorted(label_list)

        # Map label → index
        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Parse JSON
        json_name = os.path.splitext(img_name)[0] + '.json'
        json_path = os.path.join(self.annotation_dir, json_name)
        with open(json_path, 'r') as f:
            data = json.load(f)[0]

        # Multi-label encoding
        label_vector = torch.zeros(len(self.labels))
        for ann in data['annotations']:
            label = ann['label']
            if label in self.label2idx:
                label_vector[self.label2idx[label]] = 1

        if self.transform:
            image = self.transform(image)

        return image, label_vector

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = AircraftFuselageDataset(
        image_dir='../Code/Dataset/Aircraft_Fuselage_DET2023/aircraft_fuselage_coco/images',
        annotation_dir='../Code/Dataset/Aircraft_Fuselage_DET2023/aircraft_fuselage_coco/annotations',
        transform=transform
    )

    img, label_vector = dataset[0]
    # Convert label vector to names
    active_indices = torch.nonzero(label_vector).squeeze().tolist()
    # Handle single label case
    if isinstance(active_indices, int):
        active_indices = [active_indices]
    label_names = [dataset.labels[i] for i in active_indices]

    print(f"Image shape: {img.shape}")
    print(f"Multi-label vector: {label_vector }")
    print("Label names:", label_names)