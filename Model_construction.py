# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.8_mlp
# 4.1节
# 注释：黄文俊
# E-mail：hurri_cane@qq.com

import torch
from torch import nn

class MLP(nn.Module):
    # 生命MLP类继承自nn.Module类
    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):
        '''
        **kwargs表示函数接收可变长度的关键字参数字典作为函数的输入。
        当我们需要函数接收带关键字的参数作为输入的时候，应当使用**kwargs。
        我们可以通过以下这个例子来进一步理解**kwargs
        def test_kwargs(**kwargs):
            if kwargs is not None:
                for key, value in kwargs.iteritems():
                    print("{} = {}".format(key,value))
        test_kwargs(name="python", value="5")
        以上代码的执行结果如下：
        name = python
        value = 5
        '''

        # 调用MLP父类Module的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)
        # super(Net, self).init()的含义为：
        # 调用Net的父类，此例中为MLP的父类Module构造函数
        # 将MLP类的对象（self）转换为MLP父类Module的对象
        # 然后“被转换”的类Module对象调用自己的__init__函数.

        self.hidden = nn.Linear(784, 256) # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)  # 输出层


    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

X = torch.rand(2, 784)
net = MLP()
print(net)
print(net(X))



# 4.1.2.1 Sequential类
class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict
            # isinstance(object,classinfo)
            # 这个函数的功能简单的说就是判断object实例是否是classinfo类型的，如果是则返回TRUE,否则返回FALSE
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                # 将传入参数args索引赋给idx，数据赋值给module
                # 通过idx, module便可确定网络中第几层是什么网络连接
                self.add_module(str(idx), module)
    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成员
        for module in self._modules.values():
            input = module(input)
        return input


net = MySequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
        )
print(net)
print(net(X))



# 4.1.2.2 ModuleList类

net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) # # 类似List的append操作
print(net[-1])  # 类似List的索引访问
print(net)
# net(torch.zeros(1, 784)) # 会报NotImplementedError
# 报错的原因是：
# ModuleList仅仅是一个储存各种模块的列表，这些模块之间没有联系也没有顺序
# 所以不用保证相邻层的输入输出维度匹配，所以没有实现forward（前向传播）功能，导致报错


# ModuleDict类
# ModuleDict接收一个子模块的字典作为输入, 然后也可以类似字典那样进行添加访问操作:
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
print(net.output)
print(net)
# net(torch.zeros(1, 784)) # 会报NotImplementedError


#4.1.3 构造复杂的模型
class FancyMLP(nn.Module):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)

        self.rand_weight = torch.rand((20, 20), requires_grad=False) # 不可训练参数（常数参数）
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        # 使用创建的常数参数，以及nn.functional中的relu函数和mm函数
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        # 复用全连接层。等价于两个全连接层共享参数
        x = self.linear(x)
        # 控制流，这里我们需要调用item函数来返回标量进行比较
        # print(x.norm().item())
        while x.norm().item() > 1:
            # x.norm()为求X的范数
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()

X = torch.rand(2, 20)
net = FancyMLP()
print(net)
print(net(X))



print("*"*30)