import numpy as np

import torch

from dataset import PlushieTrainDataset
from model import SiameseNetwork
from transforms import Transforms
from utils import DeviceDataLoader, accuracy, get_default_device, to_device
import torch.nn as nn
import torch.nn.functional as F


def loss_batch(model, loss_func, anchor, image, label, opt=None, metric=None): # Update model weights and return metrics given xb, yb, model
    output1, output2 = model(anchor, image)
    loss = loss_func(output1, output2, label)
    
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
        
    metric_result = None
    if metric is not None:
        metric_result = metric(output1, output2, label)
        
    return loss, len(anchor), metric_result


def fit(epochs, model, loss_func, train_dl, val_dl, opt_func=torch.optim.SGD, lr=0.01, metric=None):
    train_losses, val_losses, val_metrics = [] , [], []
    
    opt = opt_func(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs+1):
        model.train() # Setting for pytorch - training mode
        for anchor,image,label in train_dl:
            train_loss, _, _ = loss_batch(model, loss_func, anchor, image, label, opt) # update weights
            
        model.eval() # Setting - eval mode
        val_loss, total, val_metric = evaluate(model, loss_func, val_dl, metric)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
        
        if metric is None:
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}".format(
            epoch, train_loss, val_loss))
        else:
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}".format(
            epoch, train_loss, val_loss, metric.__name__, val_metric))
            
    return train_losses, val_losses, val_metrics


def evaluate(model, loss_func, val_dl, metric=None):
    with torch.no_grad():
        results = [loss_batch(model, loss_func, anchor, image, label, metric=metric) for anchor, image, label in val_dl]
        
        losses, nums, metrics = zip(*results)
        total = np.sum(nums)
        
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
            
        return avg_loss, total, avg_metric


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
    #train_filepath = ""
    train_img_dir = "/content/drive/MyDrive/Brainhack/ReID/datasets/reID_dataset"
    #val_filepath = ""
    val_img_dir = "/content/drive/MyDrive/Brainhack/ReID/datasets/reID_dataset"
    train_bs = 32
    test_bs = 5
    num_epochs = 5
    lr = 0.005
    val_ratio = 0.2
    
    torch.autograd.set_detect_anomaly(True)
    
    transform = Transforms()
    train_dataset = PlushieTrainDataset(img_dir=train_img_dir, transform=transform)
    valid_dataset = PlushieTrainDataset(img_dir=val_img_dir, transform=transform)
    network = SiameseNetwork()
    

    print("The length of Train set is {}".format(len(train_dataset)))
    print("The length of Valid set is {}".format(len(valid_dataset)))

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=train_bs, shuffle=True, num_workers=4)
    val_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=test_bs, shuffle=True, num_workers=4)

    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    to_device(network, device)
  
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam


    train_losses, val_losses, val_metrics = fit(num_epochs, network, criterion, 
                                            train_dl, val_dl, optimizer, lr, accuracy)

    torch.save(network.state_dict(), 'model.pth')

    




if __name__ == "__main__":
    main()
