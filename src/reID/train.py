import numpy as np

import torch

from dataset import SiameseDataset
from model import SiameseNetwork
# from transforms import Transforms
# from utils import DeviceDataLoader, accuracy, get_default_device, to_device
import torch.nn as nn
import torch.nn.functional as F
# from test import predict_image
import torch as tt
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt




class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

def main():
    # Declare Siamese Network
    net = SiameseNetwork().cuda()
    # Decalre Loss Function
    criterion = ContrastiveLoss()
    # Declare Optimizer
    optimizer = tt.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)
    #load training data
    train_dir = '/content/drive/MyDrive/Brainhack/ReID/datasets/reducedDataset'
    train_dataset = SiameseDataset(train_dir, transform=transforms.Compose([transforms.Resize((105,105)),
                                                                            transforms.ToTensor()
                                                                            ]))
    train_dataloader = DataLoader(train_dataset, num_workers=4,batch_size=32,shuffle=True)
    #train the model
    def train(epochs):
        loss=[] 
        counter=[]
        iteration_number = 0
        for epoch in range(1,epochs):
            for i, data in enumerate(train_dataloader,0):
                img0, img1 , label = data
                img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
                optimizer.zero_grad()
                output1,output2 = net(img0,img1)
                loss_contrastive = criterion(output1,output2,label)
                loss_contrastive.backward()
                optimizer.step()    
            print("Epoch {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
            iteration_number += 10
            counter.append(iteration_number)
            loss.append(loss_contrastive.item())
        plt.plot(counter, loss)
        return net
    #set the device to cuda
    device = torch.device('cuda' if tt.cuda.is_available() else 'cpu')
    print(f'device used {device}')
    model = train(10) #10 epochs
    torch.save(model.state_dict(), "reid_model.pt")
    print("Model Saved Successfully") 

if __name__ == '__main__':
  main()