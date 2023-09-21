'''
Description: 
Author: Guan Xiongjun
Date: 2022-08-24 09:35:51
LastEditTime: 2022-08-24 09:36:04
'''
import cv2

def localEqualHist(img):
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(32,32))
    dst = clahe.apply(img)
    return dst