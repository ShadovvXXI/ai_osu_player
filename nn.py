import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class OsuImageDataset(Dataset):
    def __init__(self, song, transform=None, target_transform=None):
        self.song = song
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.song)

    def __getitem__(self, idx):
        # TODO : Сделать основную реализацию датасета
        image = 0
        label = 0
        # TODO : ____________________________________
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class OsuNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pool_part = nn.Sequential(
            nn.Conv2d(5, 32, 3, 2),
            nn.Conv2d(32, 64, 3, 2),
            nn.MaxPool2d(2, 2),
            nn.MaxPool2d(2, 2)
        )
        self.linear_part = nn.Sequential(
            nn.Linear(8640, 100),
            nn.Linear(100, 2),
        )

    def forward(self, x):
        x = self.conv_pool_part(x)
        x = nn.Flatten()(x)
        x = self.linear_part(x)
        return x

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = OsuNeuralNetwork().to(device)