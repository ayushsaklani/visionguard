import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import scipy.io
from utils import PedestrianAttributeDataset,show_grid,get_device,get_dataloader
from model import VisionGuard
import os 


def train(model,dataloader,criterion,optimizer,scheduler=None,num_epochs=100,device ="cpu",out="outputs",tf_logs="tf_logs"):
    writer = SummaryWriter(tf_logs)
    model = model.to(device)
    total_loss = 0
    for epoch in range(num_epochs):
        #train
        model.train()
        loop = tqdm(dataloader["train"],desc=f' Training Epoch {epoch + 1}/{num_epochs}', unit='batch')
        for batch_idx,data in enumerate(loop):
            optimizer.zero_grad()
            images,labels = data
            images = images.to(device,dtype = torch.float)
            labels = labels.to(device,dtype = torch.float)
            
            out = model(images)
            loss = criterion(out,labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Log loss to TensorBoard
            iteration = epoch * len(loop) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), iteration)
            loop.set_postfix(loss = loss.item())
        average_loss = total_loss / len(train_dataloader)
        writer.add_scalar("Train/AvgLoss", average_loss, global_step=epoch)
        loop.set_postfix(loss = loss.item(),avg_loss = average_loss)
        
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            loop = tqdm(dataloader["val"],desc=f' Validation', unit='image')
            for batch_idx,data in enumerate(loop):
                images,labels = data
                images = images.to(device,dtype = torch.float)
                labels = labels.to(device,dtype = torch.float)
                
                out = model(images)
                loss = criterion(out,labels)
                
                loop.set_postfix(loss = loss.item())
                
                val_loss += loss.item()
            average_loss = val_loss / len(loop)
            writer.add_scalar("Val/Loss", average_loss, global_step=epoch)
            loop.set_postfix(avg_loss = average_loss)
            print(f"Validation Loss: {average_loss}")
                
                
            
        
        if not os.path.exists(out):
            os.mkdir(out)
        torch.save(model.state_dict(),os.path.join(out,'swin_transformer_model.pth'))
    
    writer.close()
    

if __name__=="__main__":
    train_dataloader = get_dataloader(annotation_path="../gcs/pa-100k/annotation/annotation.mat",image_folder="../gcs/pa-100k/release_data/",split = "Train",batch_size =4)
    val_dataloader = get_dataloader(annotation_path="../gcs/pa-100k/annotation/annotation.mat",image_folder="../gcs/pa-100k/release_data/",split = "Val",batch_size =1)
    dataloader ={"train":train_dataloader,
                "Val":val_dataloader}
    
    model = VisionGuard()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay = 1e-5)
    
    train(model,dataloader,criterion,optimizer,device=get_device())
    