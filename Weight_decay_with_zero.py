# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.12_weight-decay?id=_3123-%e4%bb%8e%e9%9b%b6%e5%bc%80%e5%a7%8b%e5%ae%9e%e7%8e%b0
# 3.12 权重衰减从零实现
#注释：黄文俊
#邮箱：hurri_cane@qq.com
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l


#定义总体样本数据
n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

#train_features为训练时的输入特征，其包含n_train个样本，即20个
#test_features为检测时的输入特征，其包含n_test个样本，即100个
#可以看到训练样本远少于测试样本，这样容易导致过拟合，下面会通过权重衰减的方法来抑制过拟合
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]


#batch_size为小批量读取的数量
batch_size, num_epochs, lr = 1, 100, 0.003
#定义网络和损失函数
net, loss = d2l.linreg, d2l.squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
"""
    torch.utils.data.DataLoader()解释：
    批训练，把数据变成一小批一小批数据进行训练。
    DataLoader就是用来包装所使用的数据，每次抛出一批数据
    从数据库(dataset)中每次抽出batch size个(1个)样本
"""

#从零开始实现权重衰减
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

def l2_penalty(w):
    # print((w**2).sum() / 2)
    return (w**2).sum() / 2


def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    # times_sum = 0     #记录次数，Debug时用
    for _ in range(num_epochs):
        for X, y in train_iter:
            # times_sum += 1
            # l2_penalty(w)为添加了L2范数惩罚项
            # net(X, w, b)为求X×w+b，即代入参数求模型输出
            # loss求的是模型输出和训练的label（标签）之间的方差
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        # print(times_sum)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())
        #上述循环for _ in range(num_epochs):走完后
        #train_ls的长度为100；test_ls长度也为100
        #之所以都为100，是因为我们设定的num_epochs（迭代次数）为100
        #train_ls和test_ls每个元素表示，在一个迭代后，训练集和测试集的模型偏差大小。
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])

    print('L2 norm of w:', w.norm().item())


# fit_and_plot(lambd=0)

fit_and_plot(lambd=3)



plt.show()



print("*"*30)