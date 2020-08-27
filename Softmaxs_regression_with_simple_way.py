# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.6_softmax-regression-scratch
# 3.7节
# 注释：黄文俊
# E-mail：hurri_cane@qq.com
import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
from collections import OrderedDict


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

# class LinearNet(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super(LinearNet, self).__init__()
#         self.linear = nn.Linear(num_inputs, num_outputs)
#     def forward(self, x): # x shape: (batch, 1, 28, 28)
#         y = self.linear(x.view(x.shape[0], -1))
#         return y
# net = LinearNet(num_inputs, num_outputs)
# print(net) # 使用print可以打印出网络的结构



# 本函数已保存在d2lzh_pytorch包中方便以后使用
class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)

#搭建网络
#搭建网络有多种方法：见3.3.3 线性回归的简洁实现中的定义模型一节
#链接：https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.3_linear-regression-pytorch?id=_333-%e5%ae%9a%e4%b9%89%e6%a8%a1%e5%9e%8b
net = nn.Sequential(
    # 写法一
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    # 写法三
    OrderedDict([
        ('flatten', FlattenLayer()),
        #FlattenLayer为展平层，将输入进来的特征进行展平，如输入28*28的图像展平为1*784
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

init.normal_(net.linear.weight, mean=0, std=0.01)   #设置网络中的线性层net.linear的权重值为均值为0标准差为0.01的正态分布（Normal distribution）

init.constant_(net.linear.bias, val=0)              #设置网络中的线性层net.linear的误差值为常数（constant）0

loss = nn.CrossEntropyLoss()                        #调用模块定义交叉熵函数

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)#指定学习率为0.1的小批量随机梯度下降（SGD）为优化算法

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)


print("*"*30)
