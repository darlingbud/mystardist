import argparse
from ast import arg
from glob import glob
from this import d
import matplotlib.pyplot as plt

from dataloader import getDataLoaders
plt.ion()
import imp
from multiprocessing.sharedctypes import Value
from operator import mod
import os
from statistics import mode
from tkinter import image_names
from unittest.loader import VALID_MODULE_NAME
from skimage import io
from stardist import star_dist,edt_prob,non_maximum_suppression,polygons_to_label
from zmq import device
join = os.path.join
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from unet import UNetStar
from distance_loss import L1_BCELoss
import monai
from monai.data import PILReader,Dataset
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    SpatialPadd,
    RandSpatialCropd,
    # RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    RandAxisFlipd,
    RandZoomd,
    RandGaussianNoised,
    # RandShiftIntensityd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd,    
    EnsureTyped,
    EnsureType,
)
from monai.visualize import plot_2d_or_3d_image
from datetime import datetime
monai.config.print_config()
print('Successfully import all requirments!')

def show_pic(writer,pic,label,val_prob,pre,step,num):
    pic=pic.to('cpu')
    label=label.to('cpu')
    pic=pic.squeeze(0).permute(1,2,0)
    dists,probs=pre[0].to('cpu'),pre[1].to('cpu')
    val_prob=val_prob.to('cpu').squeeze(0).squeeze(0)
    label=label.squeeze(0)
    dists=dists.squeeze(0).permute(1,2,0)
    probs=probs.squeeze(0).squeeze(0)
    dists=dists.detach().numpy()
    probs=probs.detach().numpy()
    points,score,dists=non_maximum_suppression(dists,probs,prob_thresh=0.5,nms_thresh=0.5)
    img_shape=[pic.shape[0],pic.shape[1]]
    pro_star_label=polygons_to_label(dists,points,shape=img_shape) #shape is the img shape
    fig=plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(pic)
    plt.subplot(1,3,2)
    plt.imshow(val_prob)
    plt.subplot(1,3,3)
    plt.imshow(probs)
    writer.add_figure(tag=f'show step{step}/num{num}',figure=fig)
    


def main():
    parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation',add_help=False)
    parser.add_argument('--work_dir',default='./stardist/work_dir')
    parser.add_argument('--data_path',default='./stardist/data')
    parser.add_argument('--seed',default=2022,type=int)
    parser.add_argument('--radial',default=32)
    parser.add_argument('--lr',default=6e-4)
    parser.add_argument('--max_epochs',default=100)
    parser.add_argument('--epoch_tolerance',default=5)
    parser.add_argument('--val_interval',default=2)
    parser.add_argument('--input_size',default=256)
    parser.add_argument('--max_dist',default=65)
    args=parser.parse_args()
    
    #os.makedirs(model_path,exist_ok=True)
    run_id=datetime.now().strftime("%Y%m%d_%H%M")
    model_path=join(args.work_dir,run_id)

    train_transforms = Compose(
        [
            #LoadImaged(keys=["img"], reader=PILReader, dtype=np.uint8), # image three channels (H, W, 3);
            #AddChanneld(keys=["prob"], allow_missing_keys=True), # prob: (1, H, W)
            #AsChannelFirstd(keys=['img'], channel_dim=-1, allow_missing_keys=True), # image: (3, H, W)
           # AsChannelFirstd(keys=['dist'], channel_dim=-1, allow_missing_keys=True), # dist: (32, H, W)
            ScaleIntensityd(keys=["img"], allow_missing_keys=True), # Do not scale label
            SpatialPadd(keys=["img","prob","dist"], spatial_size=args.input_size),
            RandSpatialCropd(keys=["img","prob","dist"], roi_size=args.input_size, random_size=False),
            RandAxisFlipd(keys=["img", "prob","dist"], prob=0.5),
            RandRotate90d(keys=["img", "prob","dist"], prob=0.5, spatial_axes=[0, 1]),
            # # # intensity transform 
            RandGaussianNoised(keys=['img'], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["img"], prob=0.25, gamma=(1,2)),
            RandGaussianSmoothd(keys=["img"], prob=0.25, sigma_x=(1,2)),
            RandHistogramShiftd(keys=["img"], prob=0.25, num_control_points=3),
            #RandZoomd(keys=["img", "label"], prob=0.15, min_zoom=0.8, max_zoom=1.5, mode=['area', 'nearest']),
            EnsureTyped(keys=["img", "prob","dist"]),
        ]
    )
    val_transforms = Compose(
        [
            ScaleIntensityd(keys=["img"], allow_missing_keys=True),
            # AsDiscreted(keys=['label'], to_onehot=3),
            EnsureTyped(keys=["img", "dist","prob"]),
        ]
    )
    train_loader,val_loader=getDataLoaders(args.data_path,args.radial,args.max_dist,train_transforms,val_transforms)

    #creat model,loss and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=UNetStar(n_channels=3,n_classes=args.radial).to(device)
    optimizer=torch.optim.Adam(model.parameters())
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,verbose=True,patience=6,eps=1e-8,threshold=1e-20)
    loss_fuc=L1_BCELoss()
    writer=SummaryWriter(model_path)
    #scheduler=ReduceLROnPlateau(optimizer, 'min',factor=0.5,verbose=True,patience=6,eps=1e-8,threshold=1e-20)

    #start training
    max_epochs=args.max_epochs
    epoch_tolerance=args.epoch_tolerance
    val_interval=args.val_interval
    best_metric=-1
    best_metric_epoch=-1
    epoch_loss_values=list()
    metric_values=list()
    writer=SummaryWriter(model_path)
    global_step=0
    for epoch in range(1,max_epochs):
        model.train()
        print(epoch)
        for step,data in enumerate(train_loader,1):
            global_step+=1
            input,train_prob,train_dist=data['img'],data['prob'],data['dist']
            input,train_prob,train_dist=input.to(device=device,dtype=torch.float32),train_prob.to(device),train_dist.to(device)
            optimizer.zero_grad()
            prediction = model(input)
            loss,l1,bce=loss_fuc(prediction,train_prob,train_dist)
            loss.backward()
            optimizer.step()
            writer.add_scalar("train_loss",(loss).item(),global_step)
            writer.add_scalar("train_bceloss",bce,global_step)
            writer.add_scalar("trian_l1loss",l1,global_step)
        
        #if epoch>0 and epoch %val_interval ==5:
        if epoch>0:
            model.eval()
            val_loss=0
            with torch.no_grad():
                for i,val_data in enumerate(val_loader,1):
                    val_images,val_pro,val_dist = val_data["img"].to(device,dtype=torch.float32)\
                        ,val_data["prob"].to(device),val_data["dist"].to(device)
                    val_predict=model(val_images)
                    loss=loss_fuc(val_predict,val_pro,val_dist)
                    val_loss+=loss
                val_loss=val_loss/val_loader.__len__()
                writer.add_scalar("val_loss",(val_loss).item(),global_step)
                scheduler.step(val_loss)
                if val_loss > best_metric:
                    best_metric = val_loss
                    best_metric_epoch = epoch
                    torch.save(model.state_dict(), join(model_path, "best_Dice_model.pth"))
                    print("saved new best metric model")

            if (epoch - best_metric_epoch) > epoch_tolerance:
                print(f"validation metric does not improve for {epoch_tolerance} epochs! current {epoch=}, {best_metric_epoch=}")
                break
               
    writer.close()
    torch.save(model.state_dict(),join(model_path,'final_model.pth')) 
          
if __name__=="__main__":
    main()