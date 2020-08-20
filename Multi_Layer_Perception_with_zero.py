# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.8_mlp
# 3.9节
#注释：黄文俊
#邮箱：hurri_cane@qq.com
import torch
import numpy as np
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

# 定义激活函数
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

# 定义模型
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2

# 定义损失函数
loss = torch.nn.CrossEntropyLoss()

num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)





print("*"*30)
