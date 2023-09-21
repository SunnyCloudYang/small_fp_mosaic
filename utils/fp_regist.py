'''
Description:  
Author: Guan Xiongjun
Date: 2022-05-08 14:53:34
LastEditTime: 2022-08-31 10:12:32
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
from utils.fp_tps import opencv_tps, tps_apply_transform, tps_module_numpy
from scipy.ndimage import shift

def match_SIFT(image1, image2):
    # sift = cv2.xfeatures2d.SIFT_create(500)
    sift = cv2.SIFT_create(500)
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    # H, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.RANSAC, 5.0)
    H, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    sx = np.sign(H[0, 0]) * np.sqrt(H[0, 0]**2 + H[0, 1]**2)
    sy = np.sign(H[1, 1]) * np.sqrt(H[1, 0]**2 + H[1, 1]**2)
    s = (sx + sy) / 2
    if abs(1 - s) > 0.1:
        raise ValueError(
            'Scaling size changes too much during rigid alignment !')
    H[0:2, 0:2] /= s
    H = np.vstack((H, np.array([[0, 0, 1]])))
    return H, src_pts, dst_pts


def match_SIFT_without_Rotate(image1, image2):
    # sift = cv2.xfeatures2d.SIFT_create(500)
    sift = cv2.SIFT_create(500)
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])

    dis = np.mean(dst_pts - src_pts, axis=0)

    H = np.array([[1, 0, dis[0]], [0, 1, dis[1]], [0, 0, 1]])

    return H, src_pts, dst_pts

def cal_image_shift(img1,img2,mask1,mask2,search_length = 5):
    h,w = img1.shape
    img1_pad = np.pad(img1, pad_width=((search_length, search_length), (search_length, search_length)), constant_values=255)
    img1_pad = np.float32(255 - img1_pad)
    mask1_pad = np.pad(mask1, pad_width=((search_length, search_length), (search_length, search_length)), constant_values=0)
    
    img2_crop = np.float32(255-img2)

    col_matrix = np.zeros((search_length*2+1,search_length*2+1))
    for i in range(search_length*2+1):
        for j in range(search_length*2+1):
            # cv2.imshow("1", mask1_pad[i:h+i,j:w+j]*255)
            # cv2.imshow("2", mask2*255)
            # cv2.waitKey(0)
            mask_blk = np.float32(mask1_pad[i:h+i,j:w+j])*mask2
            img1_blk = np.float32(img1_pad[i:h+i,j:w+j])
            img1_blk = img1_blk[mask_blk==1]
            img2_blk = img2_crop[mask_blk==1]
            # cv2.imshow("img1_crop", np.uint8(img1_blk+np.min(img1_blk)))
            # cv2.waitKey(0)
            col_matrix[i,j] = np.corrcoef(img1_blk,img2_blk)[0][1]

    x,y = np.where(col_matrix == col_matrix.max())
    return x[0]-search_length, y[0]-search_length


def mosaic_step(img_res, img_pre, img_post,img_post_enh, mask_res, mask_post, H_pre, AREA, need_corr=False):
    H, _, _ = match_SIFT_without_Rotate(img_pre, img_post)
    H = np.dot(H, H_pre)
    
    mask_post_regist = cv2.warpPerspective(mask_post.copy(),
                                  H, (img_res.shape[1], img_res.shape[0]),
                                  flags=cv2.INTER_LINEAR +
                                  cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
                                  
    mask_common = copy.deepcopy(mask_post_regist)*(1-mask_res)

    if np.sum(mask_common) < AREA:
        return img_res, mask_res,None, H, False
    
    img_post_regist = cv2.warpPerspective(img_post_enh,
                                 H, (img_res.shape[1], img_res.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)

    if need_corr is True:
        dx,dy = cal_image_shift(img_res,img_post_regist,mask_res,mask_post_regist,search_length = 10)
        img_post_regist = shift(img_post_regist,shift=[dx,dy],mode='constant',cval=255)
        mask_post_regist = shift(mask_post_regist,shift=[dx,dy],mode='constant',cval=0)

    img_post_regist_copy = copy.deepcopy(img_post_regist)
    img_res[mask_res == 0] = 0
    img_post_regist[mask_res==1] = 0
    img_res = img_post_regist + img_res

    mask_res[mask_post_regist == 1] = 1

    img_res[mask_res == 0] = 255

    return img_res, mask_res, img_post_regist_copy, H, True


def tps_regist(img_res, img_post_regist, mask_res, mask_post_regist):
    _, src_pts, dst_pts = match_SIFT_without_Rotate(img_res, img_post_regist)
    _, Hm = cv2.estimateAffinePartial2D(src_pts,
                                        dst_pts,
                                        method=cv2.RANSAC,
                                        ransacReprojThreshold=20.0)
    src_pts = src_pts.take(np.where(Hm == 1)[0], 0)
    dst_pts = dst_pts.take(np.where(Hm == 1)[0], 0)


    img_post_regist = opencv_tps(
        img_post_regist,
        dst_pts,
        src_pts,
        mode=1,
        border_value=255,
    )

    # cv2.imwrite(osp.join(res_dir,'{}.png'.format(str(idx))),img_post_regist)
    # cv2.imshow("img_res",img_res)
    # cv2.imshow("img_post", img_post)
    # cv2.waitKey(0)

    img_res[mask_res == 0] = 0
    img_post_regist[mask_res == 1] = 0
    img_post_regist[mask_post_regist == 0] = 0

    mask_res[mask_post_regist == 1] = 1

    img_res = img_post_regist + img_res

    img_res[mask_res == 0] = 255

    return img_res, mask_res


def mosaic_step_registOnly(img_res, img_pre, img_post, mask_res, mask_post,
                           H_pre, AREA):
    H, _, _ = match_SIFT_without_Rotate(img_pre, img_post)
    H = np.dot(H, H_pre)

    maskOut = cv2.warpPerspective(mask_post.copy(),
                                  H, (img_res.shape[1], img_res.shape[0]),
                                  flags=cv2.INTER_LINEAR +
                                  cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
    maskOut[mask_res == 1] = 0

    if np.sum(maskOut) < AREA:
        return None, None, H, False

    img_post_regist = cv2.warpPerspective(img_post,
                                       H, (img_res.shape[1], img_res.shape[0]),
                                       flags=cv2.INTER_LINEAR +
                                       cv2.WARP_INVERSE_MAP,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=255)
    mask_post_regist = cv2.warpPerspective(
        mask_post,
        H, (img_res.shape[1], img_res.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0)

    return img_post_regist, mask_post_regist, H, True


# def mosaic_step_tps(img_res, img_pre, img_post,mask_res, mask_post,AREA):
#     H,src_pts, dst_pts = match_SIFT_without_Rotate(img_pre, img_post)
#     H = np.dot(H,H_pre)
#     mask_area = cv2.warpPerspective(mask_post.copy(), H, (img_res.shape[1],img_res.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

#     cv2.imshow("mask",(mask_area+mask_res)*126)
#     cv2.waitKey(0)

#     mask_area = mask_area * (mask_res==0)

#     if np.sum(mask_area) < AREA:
#         return img_res,img_post,mask_res,False

#     _, Hm = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=20.0)
#     Hm = Hm.reshape((-1,))
#     src_pts = src_pts.take(np.where(Hm==1)[0],0)
#     dst_pts = dst_pts.take(np.where(Hm==1)[0],0)

#     img_post = opencv_tps(
#         img_post,
#         dst_pts,
#         src_pts,
#         mode=1,
#         border_value=255,
#     )

#     mask_post = opencv_tps(
#         mask_post,
#         dst_pts,
#         src_pts,
#         mode=1,
#         border_value=0,
#     )

#     # cv2.imshow("mask",mask_post*255)
#     # cv2.waitKey(0)

#     img_res[mask_res==0] = 0
#     img_post[mask_res==1] = 0
#     img_post[mask_post==0] = 0

#     mask_res[mask_post==1] = 1

#     img_res = img_post + img_res
#     img_res[mask_res==0] = 255

#     return img_res,img_post,mask_res,True
