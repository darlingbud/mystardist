
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from glob import glob
from tifffile import imread
from csbdeep.utils import normalize
from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label
from stardist import random_label_cmap,ray_angles
import metric
import torch
from monai.transforms import ScaleIntensity
from unet import UNetStar as UNet
from torch.utils.tensorboard import SummaryWriter

DATASET_PATH_IMAGE = './dataset/dsb2018/test/images/*.tif'
DATASET_PATH_LABEL = './dataset/dsb2018/test/masks/*.tif'
MODEL_WEIGHTS_PATH= './mystardist/trained_point/best_Dice_model100.pth'


X = sorted(glob(DATASET_PATH_IMAGE))
X = list(map(imread,X))
Y = sorted(glob(DATASET_PATH_LABEL))
Y = list(map(imread,Y))

n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
axis_norm = (0,1)   # normalize channels independently
# axis_norm = (0,1,2) # normalize channels jointly
if n_channel > 1:
    print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
N_RAYS = 32
angles = ray_angles(N_RAYS)

def plot(target_label,pred_label):
    fig=plt.figure(figsize=(16,10))
    plt.subplot(211)
    plt.imshow(target_label.squeeze(),cmap=random_label_cmap())
    plt.axis('off')
    plt.title('Ground truth')
    plt.subplot(212)
    plt.axis('off')
    plt.imshow(pred_label,cmap=random_label_cmap())
    plt.title('Predicted Label.')
    return fig
        
def predictions(model_dist,i):
    transpose=ScaleIntensity()
    img=transpose(X[i])
    input = torch.tensor(img)
    input = input.unsqueeze(0).unsqueeze(0)#unsqueeze 2 times
    dist,prob = model_dist(input)
    dist_numpy= dist.detach().cpu().numpy().squeeze()
    prob_numpy= prob.detach().cpu().numpy().squeeze()
    return dist_numpy,prob_numpy

model_dist = UNet(1,N_RAYS)
checkpoint=torch.load(MODEL_WEIGHTS_PATH)
model_dist.load_state_dict(checkpoint["model_state_dict"])
print('Distance weights loaded')
writer=SummaryWriter("./mystardist/show")
apscore_nms = []
prob_thres = 0.4
for idx,img_target in enumerate(zip(X,Y)):
    print(idx)
    image,target = img_target
    dists,probs=predictions(model_dist,idx)
    dists = np.transpose(dists,(1,2,0))
    #coord = dist_to_coord(dists)
    points,probs,dists = non_maximum_suppression(dists,probs,prob_thresh=0.3,nms_thresh=0.3)
    img_shape=[image.shape[0],image.shape[1]]
    star_label = polygons_to_label(dists,points,img_shape,probs)
    apscore_nms.append(metric.calculateAPScore(star_label,target,IOU_tau=0.5))
    writer.add_figure("pic",plot(target,star_label),idx+1)
    #plot(target,star_label) 
print('Total images')
ap_nms = sum(apscore_nms)/(len(apscore_nms))
print('AP NMS',ap_nms)
   