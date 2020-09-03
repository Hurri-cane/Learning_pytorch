# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/
# 7.7 AdaDelta算法
# 注释：黄文俊
# E-mail：hurri_cane@qq.com

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import torch

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

img = Image.open('F:/PyCharm/Learning_pytorch/data/img/catdog.jpg')
w, h = img.size # (728, 561)

d2l.set_figsize()

def display_anchors(fmap_w, fmap_h, s):
    # 前两维的取值不影响输出结果(原书这里是(1, 10, fmap_w, fmap_h), 我认为错了)
    fmap = torch.zeros((1, 10, fmap_h, fmap_w), dtype=torch.float32)

    # 平移所有锚框使均匀分布在图片上
    offset_x, offset_y = 1.0/fmap_w, 1.0/fmap_h
    anchors = d2l.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5]) + torch.tensor([offset_x/2, offset_y/2, offset_x/2, offset_y/2])
    # d2l.MultiBoxPrior函数用处：指定输入（fmap）、一组大小和一组宽高比，该函数将返回输入的所有锚框。
    '''
    这里之所以说会均匀采样，是因为在图像位置标示值中都采用了归一化，及所有图像上的位置都可以用两个0到1的数表示。
    通过
    anchors=d2l.MultiBoxPrior(fmap,sizes=s,ratios=[1,2,0.5])+torch.tensor([offset_x/2,offset_y/2,offset_x/2,offset_y/2])
    得到的Anchors是针对fmap的anchor，其形状为1，fmap的像素高宽乘积再乘上设定的锚框高宽比长度，4
    其实就是返回fmap的像素高宽乘积再乘上设定的锚框高宽比长度个锚框，每个锚框包含4个坐标，坐标值为归一化之后的值
    在后面绘制目标图像（非fmap）时，因为采用的是归一化位置大小来表示锚框位置，所以本来在fmap上紧密排列的锚框被均匀分布了
    '''

    bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)

display_anchors(fmap_w=4, fmap_h=2, s=[0.15])
# 锚框大小s = 0.15的意思是，锚框的基准长度为整个图像长度的0.15倍
plt.show()

# 特征图的高和宽分别减半，并用更大的锚框检测更大的目标。当锚框大小设0.4时，有些锚框的区域有重合
display_anchors(fmap_w=2, fmap_h=1, s=[0.4])
plt.show()

# 最后，我们将特征图的宽进一步减半至1，并将锚框大小增至0.8。此时锚框中心即图像中心
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
plt.show()

print("*"*50)
