# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.8_mlp
# 4.2节
#注释：黄文俊
#邮箱：hurri_cane@qq.com

import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化
# 此net一共三层：输入为4，输出为3的隐藏层；ReLU激活层；输入为3，输出为1输出层

print(net)
X = torch.rand(2, 4)
Y = net(X).sum()

print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, "*",param.size())

for name, param in net[2].named_parameters():
    # 访问指定层数的参数，针对此实例中的net
    # 0表示输入为4，输出为3的隐藏层
    # 1表示ReLU激活层
    # 2表示输入为3，输出为1输出层
    print(name, param.size(), type(param))

class MyModel(nn.Module):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
        self.weight1 = nn.Parameter(torch.rand(10, 20))
        self.weight2 = torch.rand(20, 20)
    def forward(self, x):
        pass

n = MyModel()
for name, param in n.named_parameters():
    # 函数named_parameters()，返回各层中参数名称和数据
    print(name,"*",param.size())

weight_0 = list(net[0].parameters())[0]
print(weight_0.data)
print(weight_0.grad) # 反向传播前梯度为None
Y.backward()
print(weight_0.grad)

for name, param in net.named_parameters():
    if 'weight' in name:
        # 对权重项进行初始化
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)

for name, param in net.named_parameters():
    if 'bias' in name:
        # 对偏差项进行初始化
        init.constant_(param, val=0)
        print(name, param.data)

# 4.2.3 自定义初始化方法
def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)





print("*"*30)