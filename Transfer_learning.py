# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/
# 9.1 图像增广
# 注释：黄文俊
# E-mail：hurri_cane@qq.com

import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = 'D:/Program/Pytorch/Datasets'
os.listdir(os.path.join(data_dir, "hotdog")) # ['train', 'test']

train_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/train'))
test_imgs = ImageFolder(os.path.join(data_dir, 'hotdog/test'))

hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
plt.show()

# 在训练时，我们先从图像中裁剪出随机大小和随机高宽比的一块随机区域，然后将该区域缩放为高和宽均为224像素的输入。
# 测试时，我们将图像的高和宽均缩放为256像素，然后从中裁剪出高和宽均为224像素的中心区域作为输入。
# 此外，我们对RGB（红、绿、蓝）三个颜色通道的数值做标准化：
# 每个数值减去该通道所有数值的平均值，再除以该通道所有数值的标准差作为输出。

# 指定RGB三个通道的均值和方差来将图像通道归一化
# image=(image-mean)/std
# mean和std分别通过[0.485, 0.456, 0.406]；[0.229, 0.224, 0.225]进行指定
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# 满足和预训练时作同样的预处理要求

train_augs = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

test_augs = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        normalize
    ])

pretrained_net = models.resnet18(pretrained=True)
print(pretrained_net.fc)
# print(pretrained_net)

# 我们应该将最后的fc从原本的1000成修改我们需要的输出类别数2
pretrained_net.fc = nn.Linear(512, 2)
print(pretrained_net.fc)
print(pretrained_net)

# 因为fc层前的参数已经在ImageNet数据集上预训练，其参数已经足够好，所以使用较小的学习率来微调fc层前的这些参数
# 而fc层因为我们更改后参数被随机初始了，所以需要较大的学习率从头训练

output_params = list(map(id, pretrained_net.fc.parameters()))
# id(object)返回的是对象的“身份证号”，唯一且不变
# map()的原型是map(function, iterable, …)，它的返回结果是一个列表。
# 参数    function:  传的是一个函数名，可以是python内置的，也可以是自定义的。 就像上面的匿名函数lambda
# 参数    iterable:   传的是一个可以迭代的对象，例如列表，元组，字符串这样的。
'''
举例说明
a=(1,2,3,4,5)
la=map(str,a)
print(la)

输出：
['1', '2', '3', '4', '5']
'''

feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

lr = 0.01
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                       lr=lr, weight_decay=0.001)

# 微调函数
def train_fine_tuning(net, optimizer, batch_size=128, num_epochs=5):
    train_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/train'), transform=train_augs),
                            batch_size, shuffle=True)
    test_iter = DataLoader(ImageFolder(os.path.join(data_dir, 'hotdog/test'), transform=test_augs),
                           batch_size)
    loss = torch.nn.CrossEntropyLoss()
    d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)


train_fine_tuning(pretrained_net, optimizer)
print("*"*50)

# 作为对比，我们定义一个相同的模型，但将它的所有模型参数都初始化为随机值。
# 由于整个模型都需要从头训练，我们可以使用较大的学习率。
scratch_net = models.resnet18(pretrained=False, num_classes=2)
lr = 0.1
optimizer = optim.SGD(scratch_net.parameters(), lr=lr, weight_decay=0.001)
train_fine_tuning(scratch_net, optimizer)


print("*"*50)
