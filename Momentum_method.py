# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/
# 7.2 梯度下降和随机梯度下降
# 注释：黄文俊
# E-mail：hurri_cane@qq.com

from matplotlib import pyplot as plt
import numpy as np
import torch
import math
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

def momentum_2d(x1, x2, v1, v2):
    v1 = gamma * v1 + eta * 0.2 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.5
# eta控制的是此次梯度在此次迭代中的权重；gamma控制的是上一次梯度在此次迭代中的权重
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
plt.show()

eta = 0.6
d2l.show_trace_2d(f_2d, d2l.train_2d(momentum_2d))
plt.show()

# 7.4.3 从零开始实现

features, labels = d2l.get_data_ch7()

def init_momentum_states():
    v_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    v_b = torch.zeros(1, dtype=torch.float32)
    return (v_w, v_b)

def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v.data = hyperparams['momentum'] * v.data + hyperparams['lr'] * p.grad.data
        p.data -= v.data

d2l.train_ch7(sgd_momentum, init_momentum_states(),
              {'lr': 0.02, 'momentum': 0.5}, features, labels)

plt.show()

# 简洁实现
d2l.train_pytorch_ch7(torch.optim.SGD, {'lr': 0.004, 'momentum': 0.9},
                    features, labels)

plt.show()



print("*"*50)