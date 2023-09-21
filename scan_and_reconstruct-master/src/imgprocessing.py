"""
@Description :   codes that process the image, including calculating the mean image and removing the background image effect
@Author      :   Xu Zhanwei
@Time        :   2021/11/02 10:58:25
"""
import numpy as np
import cv2
import pdb
import glob
import os

## 求平均图像，除掉背景图像
# 求平均图像
def get_mean_img(img_list):
  imgs = cv2.imread(img_list[0],0)
  img_mean = np.zeros(imgs.shape)
  for imgname in img_list:
    imgs = cv2.imread(imgname,0)
    imgs = imgs.astype('float')
    img_mean += imgs/len(img_list)
  img_mean = img_mean.astype('uint8')
  return img_mean



# 除掉背景图像
def remove_back_img(img_list,img_mean,bgp=None):
  for imgname in img_list:
    imgs = cv2.imread(imgname,0)
    if bgp is None:
      bgp = np.ones(imgs.shape)
    imgs = imgs.astype('float')
    imgs = imgs/img_mean
    imgs_bg_mean = (imgs*bgp).sum()/(bgp.sum())
    imgs = imgs/imgs_bg_mean*0.12*255
    cv2.imwrite(imgname,imgs.astype('uint8'))
    # pdb.set_trace()


if __name__ == "__main__":
  root_dir = r'C:\Users\iVers\Desktop\fingerprint_reconstruction_3D\newspaper'
  img_mean_path = os.path.join(root_dir,'mean.jpg')
  img_mean = cv2.imread(img_mean_path,0)
  # img_mean = get_mean_img(img_list)
  # cv2.imwrite(os.path.join(root_dir,'mean.jpg'),img_mean)
  img_dir = os.path.join(root_dir,'imgs')
  img_list = glob.glob(os.path.join(img_dir,'*'))
  img_list.sort()
  bgp_path = os.path.join(root_dir, 'foreground.png')
  bgp = cv2.imread(bgp_path,0)
  bgp = (bgp==0).astype('float')
  bgp = cv2.resize(bgp,(img_mean.shape[1],img_mean.shape[0]),interpolation= 0)
  remove_back_img(img_list,img_mean,bgp)