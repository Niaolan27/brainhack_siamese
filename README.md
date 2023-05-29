# TIL2023 CV Qualifiers Challenge

The TIL2023 CV Qualifiers code repository

## Setting up the environment

### Clone Repository

Run this command to clone the repository  

```shell
git clone https://github.com/Niaolan27/brainhack_siamese.git
```

### Object Re-Identification

Refer to `src/reID`. The directory contains the following files:

* `dataset.py` - This file converts your images into a `torch.utils.data.Dataset` class. This Dataset takes as input (image directory, transform). The output will be a Dataset class, which can be used for training or validation. 
*
* `transforms.py` - This file preprocesses your images to ensure they're ingestible by the model. The most important preprocessing step is to resize the image to a standard size before they're passed into the model.
* 
* `model.py` - This file contains the Siamese Network. The model is made up of CNNs and a Fully Connected Layer. 
* 
* `train.py` - This file contains the code to fit your model to the dataset.
* 
* `val.py` - This file lets you validate your model on a validation dataset.
* 

```  
