'''
Description: 
Author: Guan Xiongjun
Date: 2022-08-21 11:09:34
LastEditTime: 2022-08-21 11:13:08
'''
from video2imgs import video2imgs
from imgprocessing import remove_back_img
from image_warp import sequence_warp
from img_stitch import image_stitch

import os 
import cv2
import glob
import pdb

img_dir = 'D:/GuanXJ/code/small_fp_mosaic/data/test1/'
img_list = glob.glob(os.path.join(img_dir,'*'))
stitch_image = image_stitch(img_list)
cv2.imwrite(os.path.join('D:/GuanXJ/code/small_fp_mosaic/result/base/','final.jpg'),stitch_image)

