# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.12_weight-decay?id=_3123-%e4%bb%8e%e9%9b%b6%e5%bc%80%e5%a7%8b%e5%ae%9e%e7%8e%b0
# 3.12 权重衰减简洁实现
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


dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
"""
    torch.utils.data.DataLoader()解释：
    批训练，把数据变成一小批一小批数据进行训练。
    DataLoader就是用来包装所使用的数据，每次抛出一批数据
    从数据库(dataset)中每次抽出batch size个(1个)样本
"""


def fit_and_plot_pytorch(wd):
    # 对权重参数衰减。权重名称一般是以weight结尾
    net = nn.Linear(num_inputs, 1)
    nn.init.normal_(net.weight, mean=0, std=1)
    nn.init.normal_(net.bias, mean=0, std=1)
    optimizer_w = torch.optim.SGD(params=[net.weight], lr=lr, weight_decay=wd) # 对权重参数衰减
    optimizer_b = torch.optim.SGD(params=[net.bias], lr=lr)  # 不对偏差参数衰减

    train_ls, test_ls = [], []
    #定义损失函数为均方差函数
    loss = d2l.squared_loss
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y).mean()
            optimizer_w.zero_grad()
            optimizer_b.zero_grad()

            l.backward()

            # 对两个optimizer实例分别调用step函数，从而分别更新权重和偏差
            optimizer_w.step()
            optimizer_b.step()
        train_ls.append(loss(net(train_features), train_labels).mean().item())
        test_ls.append(loss(net(test_features), test_labels).mean().item())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net.weight.data.norm().item())


#fit_and_plot_pytorch(num)，其中num表示惩罚函数的lambd值，是一个超参数
fit_and_plot_pytorch(3)


plt.show()



print("*"*30)