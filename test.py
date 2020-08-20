import torch
import torch.nn as nn
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l
import torch.utils.data as Data
from torch.nn import init

def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))
def accuracy(y_hat, y):
    a = y_hat.argmax(dim=1) == y
    b = a.float()
    c = b.mean()
    res = c.item()
    print(res)
    return (y_hat.argmax(dim=1) == y).float().mean().item()



# x = torch.tensor([[10, 2, 3], [40, 5, 60],[4,80,9]])
# y = torch.tensor([[0, 1,2,2],[0, 1,2,2],[0, 1,2,2]])
# print(x)
# # print(y)
# # print(x.gather(1, y))
#
# print("#",x.argmax(dim=1))
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y = torch.LongTensor([0, 2])
# y_hat.gather(1, y.view(-1, 1))
#
# print(accuracy(y_hat, y))



# 3.8.2 激活函数
def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')

#3.8.2.1 ReLU函数
# x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
# y = x.relu()
# # xyplot(x, y, 'relu')
#
# #ReLU函数求导
# y.sum().backward()
# # xyplot(x, x.grad, 'grad of relu')
#
# #3.8.2.2 sigmoid函数
# y = x.sigmoid()
# # xyplot(x, y, 'sigmoid')
#
# #sigmoid函数求导
# x.grad.zero_()
# y.sum().backward()
# # xyplot(x, x.grad, 'grad of sigmoid')
#
# #3.8.2.3 tanh函数
# y = x.tanh()
# xyplot(x, y, 'tanh')
#
# #tanh函数求导
# x.grad.zero_()
# y.sum().backward()
# xyplot(x, x.grad, 'grad of tanh')
#
# plt.show()

#
# #3.11 模型选择、欠拟合和过拟合
# n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
# features = torch.randn((n_train + n_test, 1))
# poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
# labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
#           + true_w[2] * poly_features[:, 2] + true_b)
# labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
#
# print(features[:2], poly_features[:2], labels[:2])

# 本函数已保存在d2lzh_pytorch包中方便以后使用
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)



#3.13丢弃法
a = torch.tensor([-1,-0.5,-0.8,0,0.2,0.5,1])
print(a)
b = a.relu()
print(b)

print("*"*30)