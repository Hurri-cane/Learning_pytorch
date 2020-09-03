# 本书链接http://zh.d2l.ai/chapter_computer-vision/ssd.html#%E5%AE%9A%E4%B9%89%E6%A8%A1%E5%9E%8B
# 锚框
# 注释：黄文俊
# E-mail：hurri_cane@qq.com

from matplotlib import pyplot as plt
from mxnet import autograd, contrib, image, init, nd
from mxnet.gluon import loss as gloss, nn
import time

# 别预测层
def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3,
                     padding=1)
    # num_anchors * (num_classes + 1)

# 边界框预测层
def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)


def forward(x, block):
    block.initialize()
    return block(x)

Y1 = forward(nd.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
Y2 = forward(nd.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
# (2, 16, 10, 10)分别为(批量大小, 通道数, 高, 宽)
print((Y1.shape, Y2.shape))

print("*" * 50)