# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/
# 9.1 图像增广
# 注释：黄文俊
# E-mail：hurri_cane@qq.com

from matplotlib import pyplot as plt
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


d2l.set_figsize()
img = Image.open('F:/PyCharm/Learning_pytorch/data/img/cat1.jpg')
d2l.plt.imshow(img)

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    plt.show()
    return axes

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)

# 实现一半概率的图像水平（左右）翻转。
apply(img, torchvision.transforms.RandomHorizontalFlip())

# 实现一半概率的图像垂直（上下）翻转。
apply(img, torchvision.transforms.RandomVerticalFlip())

# 图像随机裁剪
shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
# 对上面函数参量的解释：第一个200表示将裁剪后的图像长宽分别缩放到200像素点,
#      当然也可以指定为一个元组形式，如（100,200），便是将剪切后图像长宽缩放为200*100
# scale=(0.1, 1)表示随机裁剪出一块的面积为原面积10%~100的区域
# ratio=(0.5, 2)表示该区域的宽和高之比随机取自0.5∼2
apply(img, shape_aug)

# 变换颜色
# 我们可以从4个方面改变图像的颜色：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）

# 图像的亮度随机变化为原图亮度的50%~150%即（1−0.5）~（1+0.5）。
apply(img, torchvision.transforms.ColorJitter(brightness=0.5))

# 改变色调
apply(img, torchvision.transforms.ColorJitter(hue=0.5))

# 改变对比度
apply(img, torchvision.transforms.ColorJitter(contrast=0.5))

# 改变饱和
apply(img, torchvision.transforms.ColorJitter(saturation=0.5))

#同时设置如何随机变化图像的亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue）。
color_aug = torchvision.transforms.ColorJitter(
    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
apply(img, color_aug)

# 叠加多个图像增广方法
# 将水平翻转、变换颜色、随机裁剪方法叠加起来
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
apply(img, augs)


print("*"*50)
plt.close('all')