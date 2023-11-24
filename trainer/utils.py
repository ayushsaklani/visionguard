import torch 
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms as v2
import torchvision.transforms.functional as F

import matplotlib.pyplot as plt

from PIL import Image
import scipy.io
import numpy as np
import os

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]



def load_image(path:str):
    image = Image.open(path)
    if image.mode !='RGB' or image.mode != 'RGBA': # converts grayscale image to RGB
        image = image.convert("RGB")
    return image

class PedestrianAttributeDataset(Dataset):
    def __init__(self,annotation_path:str,image_folder:str,split = "Train"):
        self.annotations =scipy.io.loadmat(annotation_path)
        self.image_folder = image_folder
        self.split = split
        self.files, self.labels = self.get_files_labels()
        # self.files = self.annotations["train_images_name"]
        # self.labels = self.annotations["train_label"]
        self.classes = self.annotations["attributes"]
        self.class2label = {x[0][0]:i for i,x in enumerate(self.classes)}
        self.label2class = {i:x[0][0] for i,x in enumerate(self.classes)}
        self.augmentations = v2.Compose([
        v2.Resize([256,192]),
        v2.Pad(10),
        # v2.RandomResizedCrop([256,192]),
        ]) 
        self.train_transforms = v2.Compose([ v2.RandomRotation(5),
        v2.RandomHorizontalFlip(0.5)])
        self.normalize = v2.Compose([v2.ToTensor(),v2.Normalize(IMG_MEAN, IMG_STD)])
    
    def get_files_labels(self):
        match self.split:
            case "Train":
                return self.annotations["train_images_name"],self.annotations["train_label"]
            case "Val":
                return self.annotations["val_images_name"],self.annotations["val_label"]
            case "Test":
                return self.annotations["test_images_name"],None
            
            
    def __getitem__(self,index):
        img = load_image(os.path.join(self.image_folder,self.files[index,0][0]))
        
        img = self.augmentations(img)
        if self.split=="Test":
            return self.normalize(img)
        if self.split == "Train":
            img = self.train_transforms(img)
        return self.normalize(img), self.labels[index]
        

    def __len__(self):
        return len(self.files)
    
    def c2l(self,x:str):
        return self.class2label[x]
    
    def l2c(self,x:int):
        return self.label2class[x]


def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    # 3, H, W
    ten = x.clone()
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1)

def show_grid(imgs):
    if not isinstance(imgs, list):
        imgs = denormalize(imgs)
        imgs = [imgs]

    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False,figsize=(15, 10))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(annotation_path="../gcs/pa-100k/annotation/annotation.mat",image_folder="../gcs/pa-100k/release_data/",split = "Train",batch_size=4):
    match split:
        case "Train":
            train_dataset = PedestrianAttributeDataset(annotation_path,image_folder,split)
            return torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        case "Val":
            val_dataset = PedestrianAttributeDataset(annotation_path,image_folder,split)
            return torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        
        case "Test":
            test_dataset = PedestrianAttributeDataset(annotation_path,image_folder,split)
            return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
        