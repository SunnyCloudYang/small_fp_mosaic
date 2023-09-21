"""
@Description :   codes that match the images and stitch according to correlation coefficient
@Author      :   Xu Zhanwei
@Time        :   2021/10/31 15:16:37
"""

import glob
import cv2
import numpy as np
import os

def cal_image_shift(img1,img2,search_length = 5):
  
  h,w = img1.shape
  img2_ga = 1-img2
  img1_ga = 1-img1
  img2crop = img2_ga[search_length:h-search_length,search_length:w-search_length]
  img2crop = img2crop-np.mean(img2crop)
  col_matrix = np.zeros((search_length*2+1,search_length*2+1))
  for i in range(search_length*2+1):
    for j in range(search_length*2+1):
      img1crop = img1_ga[i:h-search_length*2+i,j:w-search_length*2+j]
      col_matrix[i,j] = (img1crop*img2crop).sum()/(np.std(img1crop))

  x,y = np.where(col_matrix == col_matrix.max())
  return img2, x[0]-search_length, y[0]-search_length

def cal_sequence_shift(img_list):
  img_list.sort()
  length = len(img_list)
  img1 = cv2.imread(img_list[0],0)
  img1 = img1.astype('float')/255.
  x,y = 0,0
  points = [np.array((x,y))]
  for i in range(1,length):
    img2 = cv2.imread(img_list[i],0)
    img2 = img2.astype('float')/255.
    img1, shift_x, shift_y = cal_image_shift(img1,img2)
    x = x+shift_x
    y = y+shift_y
    points.append(np.array((x,y)))
  points = np.array(points)
  points[:,0] = points[:,0] - points[:,0].min()
  points[:,1] = points[:,1] - points[:,1].min()
  return points

def image_stitch(img_list):
  points = cal_sequence_shift(img_list)
  img = cv2.imread(img_list[0],0)
  h,w = img.shape
  xmax = points[:,0].max()+h
  ymax = points[:,1].max()+w
  final_image = np.zeros((xmax,ymax)).astype('float')
  count_matrix = np.zeros((xmax,ymax)).astype('float')
  length = len(img_list)
  for i in range(length):
    img2 = cv2.imread(img_list[i],0)
    img2 = img2.astype('float')/255.
    cur_x = points[i,0]
    cur_y = points[i,1]
    final_image[cur_x:cur_x+h,cur_y:cur_y+w] += img2
    count_matrix[cur_x:cur_x+h,cur_y:cur_y+w] += 1
    print(i)
  final_image = (final_image/count_matrix*255).astype('uint8')
  return final_image

if __name__ == "__main__":
  root_dir = r'C:\Users\iVers\Desktop\fingerprint_reconstruction_3D\newspaper\crop_imgs'
  img_list = glob.glob(os.path.join(root_dir,'*'))
  stitch_image = image_stitch(img_list)
  cv2.imwrite('final.jpg',stitch_image)

