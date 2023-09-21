'''
Description:  
Author: Guan Xiongjun
Date: 2022-05-08 14:53:34
LastEditTime: 2022-08-23 16:27:39
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


def mosaic_step(img_res, img_pre, img_post, mask_res, mask_post, H_pre, AREA):
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
        return img_res, mask_res, H, False

    imgOut = cv2.warpPerspective(img_post,
                                 H, (img_res.shape[1], img_res.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=255)
    imgOut = imgOut * (maskOut == 1)

    img_res[mask_res == 0] = 0
    maskOut[mask_res == 1] = 1

    imgOut = imgOut + img_res
    imgOut[maskOut == 0] = 255

    return imgOut, maskOut, H, True


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

    cv2.imwrite(osp.join(res_dir,'{}.png'.format(str(idx))),img_post_regist)
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

if __name__ == "__main__":
    img_dir = "D:/GuanXJ/code/small_fp_mosaic/data/test2/"
    res_dir = "D:/GuanXJ/code/small_fp_mosaic/result/step_test2_tps/"

    if not osp.exists(res_dir):
        os.makedirs(res_dir)

    # # set finger input index
    f_lst = os.listdir(img_dir)
    f_lst = sorted(f_lst, key=lambda x: int(x[:-4].replace('fp_', '')))

    # # initial result image settings
    img_res = cv2.imread(osp.join(img_dir, f_lst[0]), cv2.IMREAD_GRAYSCALE)
    img_res = np.pad(img_res,
                     pad_width=((320, 320), (320, 320)),
                     constant_values=255)
    mask_res = np.zeros_like(img_res)
    mask_res[320:480, 320:480] = 1
    AREA = 160 * 160 * 0.1

    # # initial first image settings
    img_pre = copy.deepcopy(img_res)
    H_pre = np.eye(3)

    # # regist step
    # idx = 0
    # for i in tqdm(range(1,len(f_lst))):
    #     img_post = cv2.imread(osp.join(img_dir,f_lst[i]), cv2.IMREAD_GRAYSCALE)
    #     img_post = np.pad(img_post, pad_width=((320, 320), (320, 320)), constant_values=255)
    #     mask_post = np.zeros_like(img_post)
    #     mask_post[325:475,325:475]=1

    #     img_res, mask_res,H_pre, CHANGE = mosaic_step(img_res, img_pre, img_post,mask_res, mask_post,H_pre, AREA)
    #     img_pre = img_post

    #     if CHANGE is True:
    #         cv2.imwrite(osp.join(res_dir,'{}.png'.format(str(idx))),img_res)
    #         idx+=1

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
    idx = 0
    for i in tqdm(range(1, len(f_lst))):
        img_post = cv2.imread(osp.join(img_dir, f_lst[i]),
                              cv2.IMREAD_GRAYSCALE)
        img_post = np.pad(img_post,
                          pad_width=((320, 320), (320, 320)),
                          constant_values=255)
        mask_post = np.zeros_like(img_post)
        mask_post[325:475, 325:475] = 1

        img_post_regist, mask_post_regist, H_pre, CHANGE = mosaic_step_registOnly(
            img_res, img_pre, img_post, mask_res, mask_post, H_pre, AREA)

        img_pre = img_post


        if CHANGE is True:
            img_res,mask_res = tps_regist(img_res,img_post_regist,mask_res,mask_post_regist)
            # cv2.imwrite(osp.join(res_dir,'{}.png'.format(str(idx))),img_res)
            idx+=1