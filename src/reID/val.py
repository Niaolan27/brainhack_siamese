from dataset import SiameseDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as tt
import torch.nn.functional as F
import cv2
import torchvision
from model import SiameseNetwork
from inference import infer
from transforms import Transforms
import gdown

def val(model):
    # check the hardware
    device = tt.device('cuda' if tt.cuda.is_available() else 'cpu')
    # print(device)

    #loading the validation data
    val_dir = '/content/drive/MyDrive/Brainhack/ReID/datasets/testDataset' # file path to validation dataset
    val_dataset = SiameseDataset(val_dir, transform=Transforms)
    val_dataloader = DataLoader(val_dataset, num_workers=4,batch_size=1,shuffle=True)

    #test the network
    count=0
    tp, fp, tn, fn = 0, 0, 0, 0 # manual implementation of P/R
    for i, data in enumerate(val_dataloader,0): 
        x0, x1 , label = data
        output1,output2 = model(x0.to(device),x1.to(device))

        pred, euclidean_distance = infer(model, x0, x1)

        if label==tt.FloatTensor([[0]]):
            label=0
        else:
            label=1
        #tabulating results
        if pred == 1:
          if pred == label:
            tp += 1
          else:
            fp += 1
        elif pred == 0:
          if pred == label:
            tn += 1
          else:
            fn += 1
        #print(type(torchvision.utils.make_grid(concat))) 
        #print(torchvision.utils.make_grid(concat).shape)   
        #cv2.imshow('test', torchvision.utils.make_grid(concat).numpy())
        print("Predicted Euclidean Distance:-",euclidean_distance.item())
        print("Actual Label:-",label)
        count=count+1
        if count ==100:
            break
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print(f'precision = {precision} recall = {recall}')

model = SiameseNetwork()
model_id = '1MSUtLHjFWsUU6KIoI59C1fl-DmpSRU27'
output = 'reid_model.pt'
gdown.download(model_id, output, quiet = False)
model.load_state_dict(tt.load('reid_model.pt')) # file path to trained model
if tt.cuda.is_available():
    print('model cuda successful')
    model.cuda()
val(model)