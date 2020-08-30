# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/
# 7.5 AdaGrad算法
# 注释：黄文俊
# E-mail：hurri_cane@qq.com

from matplotlib import pyplot as plt
import numpy as np
import torch
import math
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l


# AdaGrad算法，它根据自变量在每个维度的梯度值的大小来调整各个维度上的学习率，从而避免统一的学习率难以适应所有维度的问题
# 如果目标函数有关自变量中某个元素的偏导数一直都较大，那么该元素的学习率将下降较快

def adagrad_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6  # 前两项为自变量梯度
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))

plt.show()

eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
plt.show()

# 7.5.3 从零开始实现
features, labels = d2l.get_data_ch7()

def init_adagrad_states():
    s_w = torch.zeros((features.shape[1], 1), dtype=torch.float32)
    s_b = torch.zeros(1, dtype=torch.float32)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s.data += (p.grad.data**2)
        p.data -= hyperparams['lr'] * p.grad.data / torch.sqrt(s + eps)

d2l.train_ch7(adagrad, init_adagrad_states(), {'lr': 0.1}, features, labels)
plt.show()

# 简洁实现
d2l.train_pytorch_ch7(torch.optim.Adagrad, {'lr': 0.1}, features, labels)
plt.show()

print("*"*50)