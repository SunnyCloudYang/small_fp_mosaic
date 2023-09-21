"""
@Description :   This file is a function for correctting the image distortion implementated by pytorch
@Author      :   Xu Zhanwei
@Time        :   2021/10/30 16:28:23
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class Warper2d(nn.Module):
    def __init__(self, img_size):
        super(Warper2d, self).__init__()
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
#        img_src: [B, 1, H1, W1] (source image used for prediction, size 32)
        img_smp: [B, 1, H2, W2] (image for sampling, size 44)
        flow: [B, 2, H1, W1] flow predicted from source image pair
        """
        H,W = img_size
        self.H, self.W = img_size
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,H,W)
        yy = yy.view(1,H,W)
        self.grid = torch.cat((xx,yy),0).float() # [2, H, W]
    def create_flow(self,param):
      k1,k2,k3,x0,y0,d,l = param
      self.flow = self.grid.clone()  
      r_s = ((self.grid[0]-x0)**2+(self.grid[1]-y0)**2)/10000
      self.flow[0,...] = x0+(self.grid[0]-x0)*(1+k1*r_s+k2*r_s**2+k3*r_s**3)
      self.flow[1,...] = y0+l*(self.grid[1]-y0)*(1+k1*r_s+k2*r_s**2+k3*r_s**3)
    def forward(self, img):
#        if flow.shape[2:]!=img.shape[2:]:
#            pad = int((img.shape[2] - flow.shape[2]) / 2)
#            flow = F.pad(flow, [pad]*4, 'replicate')#max_disp=6, 32->44
        vgrid = self.flow.clone()
        vgrid = Variable(vgrid, requires_grad = False)
        img = torch.Tensor(img)[None,None,...]
        vgrid = vgrid[None,...]
        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/(self.W-1)-1.0 #max(W-1,1)
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/(self.H-1)-1.0 #max(H-1,1)
        # vgrid = 2.0*vgrid/(self.img_size-1)-1.0 #max(W-1,1)
 
        vgrid = vgrid.permute(0,2,3,1)        
        output = F.grid_sample(img, vgrid)
#        mask = Variable(torch.ones(img.size())).cuda()
#        mask = F.grid_sample(mask, vgrid)
#        
#        mask[mask<0.9999] = 0
#        mask[mask>0] = 1
        return output#*mask

import cv2
import numpy as np
import pdb
import glob
import os
def sequence_warp(img_list):       
        img = cv2.imread(img_list[0],0)
        k = [0.0158, 0.0012, 0.0001, 310.65, 260.00, 0.8298, 1.0827]
        # k = [0.0014,    0.0085,    0.0011, 320, 240, 0.7706, 1.125]
        img_size = img.shape

        Warper = Warper2d(img_size)
        Warper.create_flow(k)

        for img_name in img_list:
                img = cv2.imread(img_name,0)
                image = Warper.forward(img)
                image = np.array(image)[0,0].astype('uint8')
                
                # crop valid zreas of the images
                xmin = 180
                ymin = 140
                xmax = 400
                ymax = 370
                image = image[xmin:xmax,ymin:ymax]
                cv2.imwrite(img_name.replace('/imgs/','/undistort_imgs/'),image)
                # pdb.set_trace()
if __name__ == "__main__":
        imgs_dir = '/home/xuzhanwei/reconstruction_finger/imgs'
        img_list = glob.glob(os.path.join(imgs_dir,'*'))
        img_list.sort()
        sequence_warp(img_list)


