'''
Description:  
Author: Guan Xiongjun
Date: 2022-05-08 14:53:34
LastEditTime: 2022-08-31 10:15:18
LastEditors: Please set LastEditors
'''
import os
import os.path as osp
from pickle import TRUE
import numpy as np
from glob import glob
import cv2
import time
from tqdm import tqdm
import copy
from utils.fp_uni import localEqualHist
from utils.fp_regist import mosaic_step


if __name__ == "__main__":
    img_dir = "D:/GuanXJ/code/small_fp_mosaic/data/test1/"
    res_dir = "D:/GuanXJ/code/small_fp_mosaic/result/step_test1_corr/"

    if not osp.exists(res_dir):
        os.makedirs(res_dir)

    # # set finger input index
    f_lst = os.listdir(img_dir)
    f_lst = sorted(f_lst, key=lambda x: int(x[:-4].replace('fp_', '')))

    # # initial first and result image settings
    img_pre = cv2.imread(osp.join(img_dir, f_lst[0]), cv2.IMREAD_GRAYSCALE)
    border_mask = 255*np.ones_like(img_pre)
    border_mask[5:-5,5:-5]=0
    img_res = copy.deepcopy(img_pre)
    img_res = localEqualHist(img_res)
    img_res[border_mask==1]=255
    img_pre = np.pad(img_pre,
                     pad_width=((320, 320), (320, 320)),
                     constant_values=255)
    img_res = np.pad(img_res,
                     pad_width=((320, 320), (320, 320)),
                     constant_values=255)
    mask_res = np.zeros_like(img_res)
    mask_res[330:470, 330:470] = 1
    AREA = 160 * 160 * 0.2
    H_pre = np.eye(3)

    
    # # regist step
    idx = 0
    for i in tqdm(range(1,len(f_lst))):
        img_post = cv2.imread(osp.join(img_dir,f_lst[i]), cv2.IMREAD_GRAYSCALE)
        img_post_enh = localEqualHist(img_post)
        img_post[border_mask==1]=255
        img_post = np.pad(img_post, pad_width=((320, 320), (320, 320)), constant_values=255)
        img_post_enh = np.pad(img_post_enh, pad_width=((320, 320), (320, 320)), constant_values=255)
        mask_post = np.zeros_like(img_post)
        mask_post[330:470,330:470]=1
        # cv2.imshow('img_post',mask_res*255)
        # cv2.waitKey(0)
        img_res, mask_res, img_post_regist, H_pre, CHANGE = mosaic_step(img_res, img_pre, img_post,img_post_enh,mask_res, mask_post,H_pre, AREA,need_corr=True)
        img_pre = img_post
        if CHANGE is True:
            # img_res = localEqualHist(img_res)
            # cv2.imwrite(osp.join(res_dir,'{}.png'.format(str(idx))),img_res)
            cv2.imwrite(osp.join(res_dir,'{}.png'.format(str(idx))),img_res)
            idx+=1
            

    # # regist tps
    # idx = 0
    # for i in tqdm(range(1,len(f_lst))):
    #     img_post = cv2.imread(osp.join(img_dir,f_lst[i]), cv2.IMREAD_GRAYSCALE)
    #     img_post = np.pad(img_post, pad_width=((320, 320), (320, 320)), constant_values=255)
    #     mask_post = np.zeros_like(img_post)
    #     mask_post[325:475,325:475]=1
    #     img_res,img_post,mask_res,CHANGE = mosaic_step_tps(img_res, img_pre, img_post,mask_res, mask_post,AREA)
    #     img_pre = img_post
    #     if CHANGE is True:
    #         cv2.imwrite(osp.join(res_dir,'{}.png'.format(str(idx))),img_res)
    #         cv2.imwrite(osp.join(res_dir,'{}_m.png'.format(str(idx))),mask_res*255)
    #         idx+=1

    # # regist step
    # idx = 0
    # for i in tqdm(range(1, len(f_lst))):
    #     img_post = cv2.imread(osp.join(img_dir, f_lst[i]),
    #                           cv2.IMREAD_GRAYSCALE)
    #     img_post = np.pad(img_post,
    #                       pad_width=((320, 320), (320, 320)),
    #                       constant_values=255)
    #     mask_post = np.zeros_like(img_post)
    #     mask_post[325:475, 325:475] = 1
    #     img_post_regist, mask_post_regist, H_pre, CHANGE = mosaic_step_registOnly(
    #         img_res, img_pre, img_post, mask_res, mask_post, H_pre, AREA)
    #     img_pre = img_post
    #     if CHANGE is True:
    #         img_res,mask_res = tps_regist(img_res,img_post_regist,mask_res,mask_post_regist)
    #         # cv2.imwrite(osp.join(res_dir,'{}.png'.format(str(idx))),img_res)
    #         idx+=1