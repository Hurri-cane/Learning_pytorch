# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.6_softmax-regression-scratch
# 3.6节
#注释：黄文俊
#邮箱：hurri_cane@qq.com
from _ast import Global

import torch
import torchvision
import numpy as np
import sys
sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l
import matplotlib.pyplot as plt




def softmax(X):
    # print("X",X.size())
    X_exp = X.exp()
    # print("X_exp",X_exp[0])
    partition = X_exp.sum(dim=1, keepdim=True)
    # a = X_exp / partition
    # print(a[0])
    return X_exp / partition  # 这里应用了广播机制
    #softmax返回的每张图像被划分为不同类别的概率分布
    #比如：第一张图像：在0-9类中的概率分布可以为[0.1000, 0.1056, 0.1070, 0.0940, 0.0996, 0.1088, 0.0993, 0.0953, 0.0863,0.1042]

def net(X):
    #输入进来的X的size是[256, 1, 28, 28]
    # print("叉乘前的X",X.size())

    # view_path= X.view((-1, num_inputs))

    #X.view((-1, num_inputs))的size是[256, 784]
    # print(view_path.size())
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
    #torch.mm(X.view((-1, num_inputs)), W) + b求出来的结果size为:[256, 10]
    #其表征的意义是：输入的256张图像为各自分类（一共10种）的概率
'''
torch.mm(X.view((-1, num_inputs)), W) + b解读：
X是一个由256个28*28的图像矩阵构成的张量
将X转换为

'''

#交叉熵损失函数
def cross_entropy(y_hat, y):
    #y是真是类别分布size为[256]，y_hat是每个类别的预测概率分布[256, 10]
    #y_hat.gather(1, y.view(-1, 1))表示通过y来索引y_hat对于的概率并转换为列为1的矩阵（即竖直排列）
    #以y的第一个元素为例，假设值为5，则对应该图像真是分类为5类，则提取y_hat中类为5的概率
    #因为y_hat中概率分布都是从0-9依次排列的，所以y_hat中类为5的概率即是y_hat中第一个图像对于的第5个元素的值（注意python中顺序是从0开始算的）
    # print(y_hat.gather(1, y.view(-1, 1)).size())
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

#分类准确率函数
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:      #获得图像矩阵；y获得标签值
        # view_path = net(X)
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()     #acc_sum为正确分类的个数之和
        n += y.shape[0]
    return acc_sum / n

# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    # 调试时计算次数
    times_sum = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            #X的size为[256, 1, 28, 28];y的size为[256]
            # print(y.size())

            y_hat = net(X)
            l = loss(y_hat, y).sum()
            '''
            损失函数l为交叉熵函数
            最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率。
            loss(y_hat, y)返回的是每张图像的交叉熵值，为了反映整体情况需要对其求和“.sum() ”
            '''
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            times_sum += 1
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到


            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
        print(times_sum)


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
'''
train_iter, test_iter都是从fashion_mnist数据库中读取来的小批量（256个）数据集
以train_iter为例，其第一个元素train_iter[0]便包含图像和标签信息,可以用feature, label = train_iter[0]来分别赋给feature和label
feature的大小为[深度*高度*宽度]，如：1*28*28；label常以整型数字存在，不同的数字表示所属不同的标签
'''
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
# W 的size为[784,10]
# print(W.size())
b = torch.zeros(num_outputs, dtype=torch.float)

#打开模型参数梯度
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

print(evaluate_accuracy(test_iter, net))


num_epochs, lr = 5, 0.1
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)

print("*"*30)
#训练完成后，现在就可以演示如何对图像进行分类了。给定一系列图像（第三行图像输出），我们比较一下它们的真实标签（第一行文本输出）和模型预测结果（第二行文本输出）。
X, y = iter(test_iter).next()
#不断使⽤next()函数来获取test_iter的下⼀条数据

true_labels = d2l.get_fashion_mnist_labels(y.detach().numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).detach().numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])











print("*"*30)
