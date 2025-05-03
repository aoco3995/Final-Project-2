from config import *
import torch
import torch.nn as nn
import torch.nn.functional as F

# CNN Model
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