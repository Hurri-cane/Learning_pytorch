# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/
# 4.5 读取和存储
# 注释：黄文俊
# E-mail：hurri_cane@qq.com

import torch
from torch import nn

x = torch.ones(3)
torch.save(x, 'Read&Write/x.pt')

x2 = torch.load('Read&Write/x.pt')
print(x2)

y = torch.zeros(4)
torch.save([x, y], 'Read&Write/x.pt')

xy_list = torch.load('Read&Write/x.pt')
print(xy_list)


# 4.5.2 读写模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2, 1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP()
print(net)
print(net.state_dict())
# state_dict是一个从参数名称隐射到参数Tesnor的字典对象,即可以以字典形式返回net中不同层的参数值

optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(optimizer.state_dict())
# 优化器(optim)也有一个state_dict，其中包含关于优化器状态以及所使用的超参数的信息。


# 4.5.2.2 保存和加载模型
'''
1. 保存和加载state_dict(推荐方式)
保存：
torch.save(model.state_dict(), PATH) # 推荐的文件后缀名是pt或pth

加载：
model = TheModelClass(*args, **kwargs)
model.load_state_dict(torch.load(PATH))

2. 保存和加载整个模型
保存：torch.save(model, PATH)
加载：model = torch.load(PATH)

'''

X = torch.randn(2, 3)
Y = net(X)

PATH = "./Read&Write/net.pt"
torch.save(net.state_dict(), PATH)

net2 = MLP()
net2.load_state_dict(torch.load(PATH))
Y2 = net2(X)
print(Y2 == Y)







print("*"*30)