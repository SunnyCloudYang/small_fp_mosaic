'''
Description: 
Author: Guan Xiongjun
Date: 2022-08-09 16:28:04
LastEditTime: 2022-08-23 16:57:34
'''
import imageio
import os
import os.path as osp


def img2gif(img_dir, gif_path, duration):
    """

    :param img_dir: 包含图片的文件夹
    :param gif_path: 输出的gif的路径
    :param duration: 每张图片切换的时间间隔，与fps的关系：duration = 1 / fps
    :return:
    """
    frames = []
    f_lst = os.listdir(img_dir)
    f_lst = sorted(f_lst,key=lambda x: int(x[:-4].replace('fp_','')))
    for idx in f_lst:
        img = osp.join(img_dir, idx)
        frames.append(imageio.imread(img))

    imageio.mimsave(gif_path, frames, duration=duration)
    print('Finish changing!')


if __name__ == '__main__':
    img_dir = 'D:/GuanXJ/Datasets/small_fp/test2/'
    gif_path = osp.join('D:/GuanXJ/code/small_fp_mosaic/result/gif/test2.gif')
    img2gif(img_dir=img_dir, gif_path=gif_path, duration=0.3)
