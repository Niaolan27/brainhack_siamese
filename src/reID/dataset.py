import os
import matplotlib.pyplot as plt

from PIL import Image
import cv2

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from transforms import Transforms

import numpy as np

class PlushieTrainDataset(Dataset):
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

        return img1, img2, siamese_label


class PlushieTrainDataset2(Dataset):
    
    def __init__(self, filepath, img_dir, transform=None):
        self.samples = []
        self.img_dir = img_dir
        self.transform = transform

        with open(filepath, 'r') as f:
            self.samples = [line.strip() for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        line = self.samples[i].split()
        if len(line) == 3:
            anchor_name, anchor_num, img_num = line
            #john 001 002
            img_name = anchor_name
            is_same = 1
        elif len(line) == 4:
            anchor_name, anchor_num, img_name, img_num = line
            is_same = 0
            # john jane 001 001
        else:
            print(len(line), line)
            raise Exception("Shouldn't be here")
        
        anchor = cv2.imread(os.path.join(self.img_dir, str(anchor_name), f"{anchor_name}_{anchor_num}.png"))
        img = cv2.imread(os.path.join(self.img_dir, img_name, f"{img_name}_{img_num}.png"))
        
        if self.transform:
            anchor = self.transform(anchor)
            img = self.transform(img)

        return anchor, img, is_same



def main():
    t = Transforms()
    #filepath = ""
    img_dir = "/content/drive/MyDrive/Brainhack/ReID/datasets/reID_dataset"
    d = PlushieTrainDataset(img_dir=img_dir, transform=t)
    
    e = d[0]
    axs = plt.figure(figsize=(9, 9)).subplots(1, 2)
    plt.title(e[2])
    axs[0].imshow(e[0].permute(1,2,0))
    axs[1].imshow(e[1].permute(1,2,0))

if __name__ == "__main__":
    main()
