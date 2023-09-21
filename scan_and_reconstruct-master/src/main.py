from video2imgs import video2imgs
from imgprocessing import remove_back_img
from image_warp import sequence_warp
from img_stitch import image_stitch

import os 
import cv2
import glob
import pdb
root_dir = '/home/xuzhanwei/reconstruction_finger'
video_path = os.path.join(root_dir, 'video.mp4')
img_path = os.path.join(root_dir,'imgs/image')
video2imgs(video_path,img_path,img_size=(640,480))

img_mean_path = os.path.join(root_dir,'mean.jpg')
img_mean = cv2.imread(img_mean_path,0)
img_mean = cv2.resize(img_mean,(640,480))
img_dir = os.path.join(root_dir,'imgs')
img_list = glob.glob(os.path.join(img_dir,'*'))
img_list.sort()
bgp_path = os.path.join(root_dir, 'foreground.png')
bgp = cv2.imread(bgp_path,0)
bgp = (bgp==0).astype('float')
bgp = cv2.resize(bgp,(640,480),interpolation= 0)

remove_back_img(img_list,img_mean,bgp)

sequence_warp(img_list)

img_dir = os.path.join(root_dir,'undistort_imgs')
img_list = glob.glob(os.path.join(img_dir,'*'))
stitch_image = image_stitch(img_list)
cv2.imwrite(os.path.join(root_dir,'final.jpg'),stitch_image)

