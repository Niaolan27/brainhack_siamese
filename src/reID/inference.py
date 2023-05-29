import torch.nn.functional as F
import torch as tt


def infer(model, img, target):
    device = tt.device('cuda' if tt.cuda.is_available() else 'cpu')

    #generate the embedding vectors using loaded model
    output1,output2 = model(img.to(device),target.to(device)) 

    #calculating euclidean distance and determining if match
    euclidean_distance = F.pairwise_distance(output1, output2)
    pred = (1 if abs(euclidean_distance) < 1 else 0) #1 for suspect, 0 for non-suspect
    return pred, euclidean_distance