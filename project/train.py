import os
from pickletools import optimize
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from data import DriveDataset
from model import build_unet
from loss import DiceLoss, DiceBCELoss
from utils import seeding,create_dir, epoch_time

def train(model,loader,optimizer,loss_fn,device):
    epoch_loss=0.0

    model.train()
    for x,y in loader:
        x=x.to(device, dtype=torch.float32)
        y=y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred=model(x)
        loss=loss_fn(y_pred,y)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()

    epoch_loss=epoch_loss/len(loader)
    return epoch_loss

if __name__=="__main__":
    seeding(42)

    create_dir("files")

    train_x=sorted(glob("C:/Users/88692/Desktop/project/new_data/train/image/*"))[:20]
    train_y=sorted(glob("C:/Users/88692/Desktop/project/new_data/train/mask/*"))[:20]

    valid_x=sorted(glob("C:/Users/88692/Desktop/project/new_data/test/image/*"))
    valid_y=sorted(glob("C:/Users/88692/Desktop/project/new_data/test/mask/*"))

    
    """Hyperparameters"""
    H=512
    W=512
    size=(H,W)
    batch_size=2
    num_epochs=50
    lr=1e-4
    checkpoint_path="files/checkpoint.pth"

    
    train_dataset=DriveDataset(train_x,train_y)
    valid_dataset=DriveDataset(valid_x,valid_y)

    train_loader=DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader=DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device=torch.device('cpu')
    model=build_unet()
    model=model.to(device)

    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=5,verbose=True)
    loss_fn=DiceBCELoss()


    for epoch in range(num_epochs):
        start_time=time.time()

        train_loss=train(model,train_loader,optimizer,loss_fn,device)