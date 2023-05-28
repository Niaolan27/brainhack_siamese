from dataset import SiameseDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as tt
import torch.nn.functional as F
import cv2
import torchvision
from model import SiameseNetwork

def test(model):
    # Load the test dataset
    device = tt.device('cuda' if tt.cuda.is_available() else 'cpu')
    print(device)
    test_dir = '/content/drive/MyDrive/Brainhack/ReID/datasets/testDataset'
    test_dataset = SiameseDataset(test_dir, transform=transforms.Compose([transforms.Resize((105,105)),
                                                                                transforms.ToTensor()
                                                                                ]))
    test_dataloader = DataLoader(test_dataset, num_workers=4,batch_size=1,shuffle=True)
    #test the network
    count=0
    tp, fp, tn, fn = 0, 0, 0, 0
    for i, data in enumerate(test_dataloader,0): 
        x0, x1 , label = data
        concat = tt.cat((x0,x1),0)
        output1,output2 = model(x0.to(device),x1.to(device))

        eucledian_distance = F.pairwise_distance(output1, output2)
        pred = (1 if abs(eucledian_distance) < 1 else 0)
        print(pred)
        print(f'label is {label.shape}')
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
        print("Predicted Eucledian Distance:-",eucledian_distance.item())
        print("Actual Label:-",label)
        count=count+1
        if count ==100:
            break
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print(f'precision = {precision} recall = {recall}')

model = SiameseNetwork()
model.load_state_dict(tt.load('reid_model.pt'))
if tt.cuda.is_available():
    print('model cuda successful')
    model.cuda()
test(model)