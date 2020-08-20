# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.8_mlp
# 3.10节
#注释：黄文俊
#邮箱：hurri_cane@qq.com

import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


num_inputs, num_outputs, num_hiddens = 784, 10, 256

#搭建网络
#搭建网络有多种方法：见3.3.3 线性回归的简洁实现中的定义模型一节
#链接：https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.3_linear-regression-pytorch?id=_333-%e5%ae%9a%e4%b9%89%e6%a8%a1%e5%9e%8b
net = nn.Sequential(
    # 写法一
    FlattenLayer(),
    #FlattenLayer为展平层，将输入进来的特征进行展平，如输入28*28的图像展平为1*784
    nn.Linear(num_inputs, num_hiddens),
    nn.ReLU(),
    nn.Linear(num_hiddens, num_outputs),
)

for params in net.parameters():
    init.normal_(params, mean=0, std=0.01)


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = torch.nn.CrossEntropyLoss()                          #调用模块定义交叉熵函数

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)       #指定学习率为0.5的小批量随机梯度下降（SGD）为优化算法

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)





print("*"*30)