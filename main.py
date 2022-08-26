import argparse
import matplotlib.pyplot as plt
from dataloader import getDataLoaders
from tqdm import tqdm
from unittest.loader import VALID_MODULE_NAME
from skimage import io
from stardist import star_dist,edt_prob,non_maximum_suppression,polygons_to_label
import os
join = os.path.join
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from unet import UNetStar
from distance_loss import L1_BCELoss
from monai.transforms import (
    Compose,
    SpatialPadd,
    RandSpatialCropd,
    RandRotate90d,
    ScaleIntensityd,
    RandAxisFlipd,
    RandZoomd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd,    
    EnsureTyped,
    EnsureType,
)
from monai.visualize import plot_2d_or_3d_image
from datetime import datetime
print('Successfully import all requirments!')

Load_SAVEPOINT=True
checkpoint_dir='./mystardist/trained_point/best_Dice_model.pth'

def main():
    parser = argparse.ArgumentParser('Baseline for Microscopy image segmentation',add_help=False)
    parser.add_argument('--work_dir',default='./stardist/work_dir')
    parser.add_argument('--data_path',default='./dataset/dsb2018')
    parser.add_argument('--seed',default=2022,type=int)
    parser.add_argument('--radial',default=32)
    parser.add_argument('--lr',default=6e-4)
    parser.add_argument('--max_epochs',default=100)
    parser.add_argument('--epoch_tolerance',default=10)
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
    model=UNetStar(n_channels=1,n_classes=args.radial).to(device)
    optimizer=torch.optim.Adam(model.parameters())
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,verbose=True,patience=6,eps=1e-8,threshold=1e-20)
    loss_fuc=L1_BCELoss()
    writer=SummaryWriter(model_path)
    #scheduler=ReduceLROnPlateau(optimizer, 'min',factor=0.5,verbose=True,patience=6,eps=1e-8,threshold=1e-20)

    #start training
    max_epochs=args.max_epochs
    epoch_tolerance=args.epoch_tolerance
    best_metric=-1
    best_metric_epoch=-1
    writer=SummaryWriter(model_path)
    global_step=0
    if(Load_SAVEPOINT):
        checkpoint=torch.load(checkpoint_dir,map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    for epoch in range(0,max_epochs):
        model.train()
        with tqdm(total=len(train_loader)*train_loader.batch_size,desc=f'Epoch {epoch+1}/{max_epochs}',unit='img') as pbar: 
            for step,data in enumerate(train_loader,1):
                global_step+=1
                input,train_prob,train_dist=data['img'],data['prob'],data['dist']
                input,train_prob,train_dist=input.to(device=device,dtype=torch.float32),train_prob.to(device),train_dist.to(device)
                optimizer.zero_grad()
                prediction = model(input)
                loss,l1,bce=loss_fuc(prediction,train_dist,train_prob)
                loss.backward()
                optimizer.step()
                writer.add_scalar("train_loss",(loss).item(),global_step)
                writer.add_scalar("train_bceloss",bce,global_step)
                writer.add_scalar("trian_l1loss",l1,global_step)
                pbar.update(input.shape[0])
         
            
            if epoch>0 and epoch%5==0:
                model.eval()
                val_loss=0
                with torch.no_grad():
                    with tqdm(total=len(val_loader),desc=f'validation loss',unit='img',leave=False) as pbar: 
                        for i,val_data in enumerate(val_loader,1):
                            val_images,val_pro,val_dist = val_data["img"].to(device,dtype=torch.float32)\
                                ,val_data["prob"].to(device),val_data["dist"].to(device)
                            val_predict=model(val_images)
                            loss,_,_=loss_fuc(val_predict,val_dist,val_pro)
                            val_loss+=loss
                            pbar.update(val_images.shape[0])
                        val_loss=val_loss/val_loader.__len__()
                        writer.add_scalar("val_loss",(val_loss).item(),global_step)
                        scheduler.step(val_loss)
                        if val_loss < best_metric:
                            best_metric = val_loss
                            best_metric_epoch = epoch
                            checkpoint={
                                            "epoch":epoch,
                                            "model_state_dict":model.state_dict(),
                                            "optimizer_state_dict":optimizer.state_dict(),
                                            "best_metric":best_metric,
                                            "best_metric_epoch":best_metric_epoch
                                        }
                            torch.save(checkpoint, join(model_path, "best_Dice_model.pth"))
                            print("saved new best metric model")

            if (epoch - best_metric_epoch) > epoch_tolerance:
                
                print(f"validation metric does not improve for {epoch_tolerance} epochs! current {epoch=}, {best_metric_epoch=}")
                break
               
    writer.close()
    torch.save(checkpoint,join(model_path,'final_model.pth')) 
          
if __name__=="__main__":
    main()