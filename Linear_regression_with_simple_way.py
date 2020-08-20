import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import sys
sys.path.append("..")
from d2lzh_pytorch import *
import torch.utils.data as Data
from torch.nn import init
import torch.optim as optim

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
for X, y in data_iter:
    print(X, y)
    break

print("*"*20)

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

print(net)

for param in net.parameters():
    print(param)

#初始化模型参数
init.normal_(net[0].weight, mean=0, std=0.01)   #修改权重参数为均值为0、标准差为0.01的正态分布
init.constant_(net[0].bias, val=0)  # 也可以直接修改bias的data: net[0].bias.data.fill_(0)

#定义损失函数
loss = nn.MSELoss()

#指定学习率为0.03的小批量随机梯度下降（SGD）为优化算法
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)



# 调整学习率
for param_group in optimizer.param_groups:
    param_group['lr'] *= 1.5 # 学习率为之前的1.5倍

print(optimizer)

loop_times = 0

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
        loop_times += 1       #记录优化次数
    print('epoch %d, loss: %f' % (epoch, l.item()))
    print(loop_times)


dense = net[0]
print("W的真实值：",true_w,'\n'"W的预测值：",dense.weight)
print("b的真实值：",true_b,'\n'"b的预测值：",dense.bias)





