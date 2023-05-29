from torchvision import transforms
import torch as tt

class Transforms:
    def __init__(self):
        #series of transformations to apply
        self.transform = tt.Compose([transforms.Resize((105,105)),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    transforms.ToTensor()]
                                    )
    def __call__(self, image):
        return self.transform(image)