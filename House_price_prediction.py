# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.8_mlp
# 3.16 实战Kaggle比赛：房价预测
# 注释：黄文俊
#E-mail：hurri_cane@qq.com

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

# print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

train_data = pd.read_csv('./House_price_prediction/train.csv')
test_data = pd.read_csv('./House_price_prediction/test.csv')

#前4个样本的前4个特征、后2个特征和标签（SalePrice）：
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])



all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# train_data.iloc[:, 1:-1]
# 表示train_data中所有的行都要，1：-1表示：列中第0列不要，最后一列不要
# 这么做的原因是，train_data中第0列为序号，最后一列为房屋价格，这些都不该是输入特征


numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
#将特征中说有数据类型不为Object的索引返回，可以理解为得到所有数据类型为数字的索引
# print(numeric_features)

# A = all_features[numeric_features]
# print(A)
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
"""
lambda理解：
lambda函数也叫匿名函数，即没有具体名称的函数，它允许快速定义单行函数，可以用在任何需要函数的地方。
g = lambda x, y, z : (x + y) ** z
print g(1,2,2)
结果为9
"""

"""
DataFrame.apply（）理解：
用途：当一个函数的参数存在于一个元组或者一个字典中时，用来间接的调用这个函数，并将元组或者字典中的参数按照顺序传递给参数
上面代码中all_features[numeric_features].apply()的意思就是：
将all_features[numeric_features]这个DataFrame的所有元素执行后面的lambda函数操作
即减去该特征的均值之后除以该特征的标准差（标准化）
"""

# 标准化后，每个数值特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)


# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
#pd.get_dummies()作用是将离散数值转成指示特征。
#假设特征MSZoning里面有两个不同的离散值RL和RM，那么这一步转换将去掉MSZoning特征
#并新加两个特征MSZoning_RL和MSZoning_RM，其值为0或1。
# print(all_features.shape)


n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values, dtype=torch.float).view(-1, 1)

loss = torch.nn.MSELoss()

def get_net(feature_num):
    net = nn.Linear(feature_num, 1)
    for param in net.parameters():
        # print(param)
        # 线性的权重param其实只有 输出数+1 个（如果偏差bias打开的话）
        # 输出数好理解，因为有多少输出就有多少个权重矩阵来计算对应不同输出的计算值
        # 而其中1对应的便是偏差bias
        nn.init.normal_(param, mean=0, std=0.01)
    return net


#评价模型的对数均方根误差函数
def log_rmse(net, features, labels):
    with torch.no_grad():
        # 将小于1的值设成1，使得取对数时数值更稳定
        # A = net(features)
        clipped_preds = torch.max(net(features), torch.tensor(1.0))
        # 这一步的意义为：将特征传入网络计算，得预测值，将预测值中小于1的值置换为1
        # 预测值在此处的现实意义是：当前网络参数下的预测房价

        rmse = torch.sqrt(loss(clipped_preds.log(), labels.log()))
    return rmse.item()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    # 这里使用了Adam优化算法
    optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net = net.float()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X.float()), y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    # 通过这个循环可以拿到每一次迭代后网络对于训练集和测试集预测值相对真实标签的偏差量（对数均方根误）
    # 并分别储存在train_ls, test_ls中
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    fold_size = X.shape[0] // k     #每一折的大小，即每个子数据集的大小
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        #根据每一折的大小构建数据集被折数i均分的索引
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
            #这一步意义为:若循环次数j和“折数”i相等，则将这部分J对应的数据赋给X_valid, y_valid（验证集）
        elif X_train is None:
            X_train, y_train = X_part, y_part
            #通过这一步可以避免j不等于i时，X_part, y_part有值可赋，并且后面可以指定以dim=0方向拼接时不出错
        else:
            X_train = torch.cat((X_train, X_part), dim=0)
            y_train = torch.cat((y_train, y_part), dim=0)
            #这一步意义为:将除折数对应的数据外的数据进行拼接，构成得到K-1个子数据作为训练集
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        # data元组中包含四个张量：
        # 从前往后依次是：通过K折交叉验证得到的  训练集特征；训练集标签；验证集特征；验证集标签
        # 其中训练用的数据集长度为K-1个子数据集长度（本案例中为1168）；验证用的数据集长度为1个子数据集长度（本案例中为292）
        net = get_net(X_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        # data前的*作用在于将data元组进行解包
        # 对于本案例来说，相当如对train（）函数传入了，四个张量，分别是：训练集特征；训练集标签；验证集特征；验证集标签

        # train_ls, valid_ls分别为第i折的训练集和测试集经过num_epochs次迭代后的网络偏差值

        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        # 将最新的偏差值进行累加，得到目前网络在训练集和测试集上的偏差之和

        if i == 0:
            d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs + 1), valid_ls,
                         ['train', 'valid'])
            plt.show()
        # 这个判断语句的意义在于：绘制出第一次K折交叉验证后当前网络的偏差随着迭代次数的增加而变化的规律

        print('fold %d, train rmse %f, valid rmse %f' % (i, train_ls[-1], valid_ls[-1]))
        #打印出第i次K折交叉验证后当前网络在训练集和测试集上的偏差
    return train_l_sum / k, valid_l_sum / k
    # 返回K折交叉验证后，当前网络在训练集和测试集上的偏差的均值


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net(train_features.shape[1])
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    plt.show()
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).detach().numpy()
    # detach(）的主要用途是将有梯度的变量变成没有梯度
    # 因为网络内的值或输出的值都有梯度，所以要想将值转换成其他类型，都需要先去掉梯度。


    ## 以下if语句下的内容为个人增补部分，其目的在于计算预测值preds与实际标签的差距
    ## E-mail：hurri_cane@qq.com
    ## 若不想运行此代码将if True:更改为
    ## if False:
    # if True:
    #     test_labels = torch.tensor(test_data.SalePrice.values, dtype=torch.float).view(-1, 1)
    #     test_rmse = torch.sqrt(loss(torch.from_numpy(preds).log(), test_labels.log()))
    #     print('test rmse  %f' % test_rmse)

    # 针对以上增加的部分，如果运行，会报错，究其原因，原来是我们从网站上下载下来的test.csv数据中就不包含有房价SalePrice项
    # 所以test_labels的赋值一步就会出错，但是若测试数据中提供了房价预测SalePrice项，运行上述代码，不会出错。


    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./House_price_prediction/submission.csv', index=False)





k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)

print('%d-fold validation: avg train rmse %f, avg valid rmse %f' % (k, train_l, valid_l))
# 打印出K折交叉验证后，当前网络在训练集和测试集上的偏差的均值

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)



print("*"*30)