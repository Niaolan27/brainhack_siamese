#preprocessing and loading the dataset
import pandas as pd
from PIL import Image
import numpy as np
import torch as tt
from torchvision.datasets import ImageFolder

class SiameseDataset():
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.dataset = ImageFolder(img_dir)
        self.labels = np.unique(self.dataset.targets)
        self.label_to_indices = {label: np.where(np.array(self.dataset.targets) == label)[0]
                                 for label in self.labels}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img1, label1 = self.dataset[index]
        same_class = np.random.randint(0, 2)
        if same_class:
            siamese_label = 1
            index2 = np.random.choice(self.label_to_indices[label1])
        else:
            siamese_label = 0
            label2 = np.random.choice(np.delete(self.labels, label1))
            index2 = np.random.choice(self.label_to_indices[label2])

        img2, _ = self.dataset[index2]

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, tt.from_numpy(np.array(siamese_label))