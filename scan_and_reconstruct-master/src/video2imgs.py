"""
@Description :   code that converts a video into image sequence of gray format
@Author      :   Xu Zhanwei 
@Time        :   2021/11/02 10:55:28
"""
import cv2
import numpy as np
import pdb
import os

# 定义保存图片函数
# image:要保存的图片名字
# addr；图片地址与相片名字的前部分
# num: 相片，名字的后缀。int 类型
def save_image(image,addr,num):
  address = addr + '%04d'%(num)+ '.jpg'
  cv2.imwrite(address,image)# 导入所需要的库


def video2imgs(video_path,img_path,img_size=None):
  videoCapture = cv2.VideoCapture(video_path)
  #读帧
  success, frame = videoCapture.read()
  i = 0
  while success :
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if img_size is not None:
      frame = cv2.resize(frame,img_size)
    save_image(frame,img_path,i)
    print('save image:',i)
    success, frame = videoCapture.read()
    i = i + 1

if __name__=="__main__":
# 读取视频文件
  root_dir = r'C:\Users\iVers\Desktop\fingerprint_reconstruction_3D\newspaper'
  video_path = os.path.join(root_dir, 'video.mp4')
  img_path = os.path.join(root_dir,'imgs\image')
  video2imgs(video_path,img_path,img_size=(640,480))