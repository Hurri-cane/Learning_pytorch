# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/
# 4.4节
# 注释：黄文俊
# E-mail：hurri_cane@qq.com



import torch
from torch import nn


# 4.4.1 不含模型参数的自定义层
# class CenteredLayer(nn.Module):
#     def __init__(self, **kwargs):
#         super(CenteredLayer, self).__init__(**kwargs)
#     def forward(self, x):
#         return x - x.mean()
#
# layer = CenteredLayer()
# print(layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)))
#
# net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
# # print(torch.rand(4, 8))       #生成4行8列的随机数（数值在0-1之间）
# y = net(torch.rand(4, 8))
# print(y.mean().item())


# 4.4.2 含模型参数的自定义层
class MyListDense(nn.Module):
    def __init__(self):
        super(MyListDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x
net1 = MyListDense()
print(net1)


class MyDictDense(nn.Module):
    def __init__(self):
        super(MyDictDense, self).__init__()
        self.params = nn.ParameterDict({
                'linear1': nn.Parameter(torch.randn(4, 4)),
                'linear2': nn.Parameter(torch.randn(4, 1))
        })
        self.params.update({'linear3': nn.Parameter(torch.randn(4, 2))}) # 新增

    def forward(self, x, choice='linear1'):
        return torch.mm(x, self.params[choice])

net2 = MyDictDense()
print(net2)

x = torch.ones(1, 4)
print(net2(x, 'linear1'))
print(net2(x, 'linear2'))
print(net2(x, 'linear3'))


net = nn.Sequential(
    MyDictDense(),
    MyListDense(),
)
print(net)
print(net(x))




print("*"*30)