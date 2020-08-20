# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.8_mlp
# 3.11 模型选择、欠拟合和过拟合
#注释：黄文俊
#邮箱：hurri_cane@qq.com

from matplotlib import pyplot as plt
import torch
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l


n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = torch.randn((n_train + n_test, 1))
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

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

num_epochs, loss = 100, torch.nn.MSELoss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):

    # 通过Linear文档可知，pytorch已经将参数初始化了，所以我们这里就不手动初始化了、
    net = torch.nn.Linear(train_features.shape[-1], 1)
    print(train_features.shape[-1])     #查看train_features.shape[-1]的值，方便理解net网络
    # net的定义是决定欠拟合还是过拟合的关键一步
    # 当用三阶多项式函数拟合（正常）时
    '''
    因为传入的train_features是poly_features参数；所以其train_features.shape[-1]等于3
    net的输入变成了3，输出torch.nn.Linear(train_features.shape[-1], 1)最后的一个1决定，所以编程输入为1，输出为1的网络
    这也是为什么这时的拟合称之为用三阶多项式函数拟合
    '''
    #当用线性函数拟合（欠拟合）时
    '''
    因为传入的train_features是features参数；所以其train_features.shape[-1]等于1
    net的输入变成了1，输出torch.nn.Linear(train_features.shape[-1], 1)最后的一个1决定，所以编程输入为1，输出为1的网络
    这也是为什么这时的拟合称之为用线性函数拟合
    '''
    #当训练样本不足（过拟合）时
    '''
    虽然传入的train_features是poly_features参数；该网络的结构也是三阶多项式
    但是输入的参数为poly_features[0:2, :], poly_features[n_train:, :],labels[0:2],labels[n_train:]
    仅含两个训练样本，但是模型负责度较高（三阶）。
    这样导致了训练样本的不足，从而导致过拟合
    '''
    batch_size = min(10, train_labels.shape[0])
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data,
          '\nbias:', net.bias.data)

#三阶多项式函数拟合（正常）
# fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
#             labels[:n_train], labels[n_train:])

#线性函数拟合（欠拟合）
# fit_and_plot(features[:n_train, :], features[n_train:, :],
#             labels[:n_train], labels[n_train:])

#训练样本不足（过拟合）
fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :],
             labels[0:2],labels[n_train:])



plt.show()

print("*"*30)