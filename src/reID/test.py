from dataset import SiameseDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch as tt
import torch.nn.functional as F
import cv2
import torchvision

def test(model):
    # Load the test dataset
    device = tt.device('cuda' if tt.cuda.is_available() else 'cpu')
    test_dir = '/content/drive/MyDrive/Brainhack/ReID/datasets/reducedDataset'
    test_dataset = SiameseDataset(test_dir, transform=transforms.Compose([transforms.Resize((105,105)),
                                                                                transforms.ToTensor()
                                                                                ]))
    test_dataloader = DataLoader(test_dataset, num_workers=4,batch_size=32,shuffle=True)
    #test the network
    count=0
    for i, data in enumerate(test_dataloader,0): 
        x0, x1 , label = data
        concat = tt.cat((x0,x1),0)
        output1,output2 = model(x0.to(device),x1.to(device))

        eucledian_distance = F.pairwise_distance(output1, output2)
            
        if label==tt.FloatTensor([[0]]):
            label="Same"
        else:
            label="Different"
            
        cv2.imshow(torchvision.utils.make_grid(concat))
        print("Predicted Eucledian Distance:-",eucledian_distance.item())
        print("Actual Label:-",label)
        count=count+1
        if count ==10:
            break