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
from mpl_toolkits import mplot3d
from PIL import Image
import os
import torchvision


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


#
# #3.13丢弃法
# a = torch.tensor([-1,-0.5,-0.8,0,0.2,0.5,1])
# print(a)
# b = a.relu()
# print(b)

#
# X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
# H, W_hh = torch.randn(3, 4), torch.randn(4, 4)
# ans1 = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
# print(ans1)
#
# ans2 = torch.matmul(torch.cat((X, H), dim=1), torch.cat((W_xh, W_hh), dim=0))
# print(ans2)



# 7.1 优化与深度学习
#
# def f(x):
#     return x * np.cos(np.pi * x)
#
# d2l.set_figsize((4.5, 2.5))
# x = np.arange(-1.0, 2.0, 0.1)
# fig1,  = d2l.plt.plot(x, f(x))
# fig1.axes.annotate('local minimum', xy=(-0.3, -0.25), xytext=(-0.77, -1.0),
#                   arrowprops=dict(arrowstyle='->'))
# fig1.axes.annotate('global minimum', xy=(1.1, -0.95), xytext=(0.6, 0.8),
#                   arrowprops=dict(arrowstyle='->'))
# '''
# matplotlib.pyplot.annotate(s, xy, *args, **kwargs)
# s：标注文本。
# xy：要标注的点，一个元组(x,y)。
# xytext：可选的，文本的位置，一个元组(x,y)。如果没有设置，默认为要标注的点的坐标。
# xycoords：可选的，点的坐标系。字符串、Artist、Transform、callable或元组。
# '''
#
# d2l.plt.xlabel('x')
# d2l.plt.ylabel('f(x)')

# # 鞍点
# x = np.arange(-2.0, 2.0, 0.1)
# fig2, = d2l.plt.plot(x, x**3)
# fig2.axes.annotate('saddle point', xy=(0, -0.2), xytext=(-0.52, -5.0),
#                   arrowprops=dict(arrowstyle='->'))
# d2l.plt.xlabel('x')
# d2l.plt.ylabel('f(x)')
#
#
#
# x, y = np.mgrid[-1: 1: 31j, -1: 1: 31j]
# # 生成一个网格，网格有两个维度，值域均是从-1到1，平均分为31份
# # 所以x,y的形状均为31*31
# # 并且值得一提的是：因为x在前y在后，所以X是纵向分布从-1到1；y是横向分布从-1到1
# z = x**2 - y**2
#
# ax = d2l.plt.figure().add_subplot(111, projection='3d')
# ax.plot_wireframe(x, y, z, **{'rstride': 2, 'cstride': 2})
# ax.plot([0], [0], [0], 'rx')
# ticks = [-1,  0, 1]
# d2l.plt.xticks(ticks)
# d2l.plt.yticks(ticks)
# ax.set_zticks(ticks)
# d2l.plt.xlabel('x')
# d2l.plt.ylabel('y')
#
# plt.show()


# # 7.4 动量法
#
# eta = 0.4 # 学习率
#
# def f_2d(x1, x2):
#     return 0.1 * x1 ** 2 + 2 * x2 ** 2
#
# def gd_2d(x1, x2, s1, s2):
#     return (x1 - eta * 0.2 * x1, x2 - eta * 4 * x2, 0, 0)
#
# d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
#
#
# # 我们试着将学习率调得稍大一点，此时自变量在竖直方向不断越过最优解并逐渐发散。
# eta = 0.6
# d2l.show_trace_2d(f_2d, d2l.train_2d(gd_2d))
#
#
# plt.show()

#
# # 8.1 命令式和符号式混合编程
# def add_str():
#     return '''
# def add(a, b):
#     return a + b
# '''
#
# def fancy_func_str():
#     return '''
# def fancy_func(a, b, c, d):
#     e = add(a, b)
#     f = add(c, d)
#     g = add(e, f)
#     return g
# '''
#
# def evoke_str():
#     return add_str() + fancy_func_str() + '''
# print(fancy_func(1, 2, 3, 4))
# '''
#
# prog = evoke_str()
# print(prog)
# y = compile(prog, '', 'exec')
# exec(y)

# 8.3 自动并行计算
# 需要有两块GPU下进行
# assert torch.cuda.device_count() >= 2
#
# class Benchmark():  # 本类已保存在d2lzh_pytorch包中方便以后使用
#     def __init__(self, prefix=None):
#         self.prefix = prefix + ' ' if prefix else ''
#
#     def __enter__(self):
#         self.start = time.time()
#
#     def __exit__(self, *args):
#         print('%stime: %.4f sec' % (self.prefix, time.time() - self.start))
# def run(x):
#     for _ in range(20000):
#         y = torch.mm(x, x)
# x_gpu1 = torch.rand(size=(100, 100), device='cuda:0')
# x_gpu2 = torch.rand(size=(100, 100), device='cuda:1')
# with Benchmark('Run on GPU1.'):
#     run(x_gpu1)
#     torch.cuda.synchronize()
#
# with Benchmark('Then run on GPU2.'):
#     run(x_gpu2)
#     torch.cuda.synchronize()
#
# with Benchmark('Run on both GPU1 and GPU2 in parallel.'):
#     run(x_gpu1)
#     run(x_gpu2)
#     torch.cuda.synchronize()


#
# # 9.3 目标检测和边界框
#
# d2l.set_figsize()
# img = Image.open('F:/PyCharm/Learning_pytorch/data/img/catdog.jpg')
# d2l.plt.imshow(img) # 加分号只显示图
# plt.show()
# # 手动绘制边界框（bounding box）
#
# # bbox是bounding box的缩写
# dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]
#
# def bbox_to_rect(bbox, color):  # 本函数已保存在d2lzh_pytorch中方便以后使用
#     # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
#     # ((左上x, 左上y), 宽, 高)
#     return d2l.plt.Rectangle(
#         xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
#         fill=False, edgecolor=color, linewidth=2)
#
# fig = d2l.plt.imshow(img)
# fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
# fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
# plt.show()
#
# Path1 = 'home'
# Path2 = 'develop'
# Path3 = 'code'
# Path10 = Path1 + Path2 + Path3
# Path20 = os.path.join(Path1,Path2,Path3)
# print ('Path10 = ',Path10)
# print ('Path20 = ',Path20)


# 9.8.2 Fast R-CNN

X = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
print(X)

rois = torch.tensor([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]], dtype=torch.float)

ans = torchvision.ops.roi_pool(X, rois, output_size=(2, 2), spatial_scale=0.1)
print(ans)
print("*"*50)