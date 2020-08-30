# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/
# 5.2 填充和步幅
# 注释：黄文俊
# E-mail：hurri_cane@qq.com

import torch
from torch import nn



# 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
    X = X.view((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.view(Y.shape[2:])  # 排除不关心的前两维：批量和通道

# 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
# padding=1表示高度方向和宽度方向均填充了，其等价于padding=（1,1）

X = torch.rand(8, 8)
print(comp_conv2d(conv2d, X).shape)

# 使用高为5、宽为3的卷积核。在高和宽两侧的填充数分别为2和1
conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)

# 令高和宽上的步幅均为2，从而使输入的高和宽减半。
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)

# 稍复杂的例子
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)
# 这个复杂的例子中，其实高和宽都没有被整除，高方向上剩了2行；宽方向上剩了1列
# 将步长改为（4,5），输出的shape还是torch.Size([2, 2])



print("*"*50)