# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.8_mlp
# 6.3 语言模型数据集（周杰伦专辑歌词）
# 注释：黄文俊
# E-mail：hurri_cane@qq.com

import torch
import random
import zipfile

with zipfile.ZipFile('F:/PyCharm/Learning_pytorch/data/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
print(corpus_chars[:40])


# 为了打印方便，我们把换行符替换成空格，然后仅使用前1万个字符来训练模型。
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:10000]
print(corpus_chars)


# 将数据集里所有不同字符取出来，然后将其逐一映射到索引来构造词典
# 这个字典char_to_idx构建的是不同文字和字符字符索引的字典，其key为文字，值为索引
# 注意：字符索引和最开始的corpus_chars并没有什么关系，这里的字符索引是重新排列的
# 并且每次运行程序时，不同字符的索引会发生变化，因为set(corpus_chars)使得重复字符随机（或许？）删去，导致顺序改变
# a = set(corpus_chars)     # set意思是创建一个集合，里面不能包含重复元素
# b = list(a)               # 将集合转化为列表

idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
vocab_size = len(char_to_idx)
print(vocab_size)


# 将训练数据集中每个字符转化为索引，并打印前20个字符及其对应的索引。
# A = char_to_idx["脉"]
corpus_indices = [char_to_idx[char] for char in corpus_chars]
sample = corpus_indices[:20]
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))
print('indices:', sample)


#  随机采样
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
    # 减1是因为输出的索引x是相应输入的索引y加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    # num_examples表示，将传入序列corpus_indices按时间步数分，最多可以分几份
    # 之所以要求这个，是为后面的
    # X = [_data(j * num_steps) for j in batch_indices]
    # Y = [_data(j * num_steps + 1) for j in batch_indices]
    # 这两句代码索引数据是不超过传入序列corpus_indices的长度


    epoch_size = num_examples // batch_size
    # epoch_size表示按照设定的读取样本数和每次时间步数，读完整个传入序列corpus_indices要读多少个循环

    example_indices = list(range(num_examples))
    random.shuffle(example_indices)     #打乱次序

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data(j * num_steps) for j in batch_indices]
        Y = [_data(j * num_steps + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.float32, device=device), torch.tensor(Y, dtype=torch.float32, device=device)


my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print("*" * 50)
    print('X: ', X, '\nY:', Y, '\n')



# 相邻采样
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size*batch_len].view(batch_size, batch_len)
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print("*" * 50)
    print('X: ', X, '\nY:', Y, '\n')


print("*"*50)