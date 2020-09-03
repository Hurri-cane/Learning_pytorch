# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/
# 锚框
# 注释：黄文俊
# E-mail：hurri_cane@qq.com

from matplotlib import pyplot as plt
import time
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

# 本函数已保存在d2lzh_pytorch中方便以后使用
def read_voc_images(root="F:/PyCharm/Learning_pytorch/data/VOCdevkit/VOC2012",
                    is_train=True, max_num=None):
    txt_fname = '%s/ImageSets/Segmentation/%s' % (
        root, 'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    if max_num is not None:
        images = images[:min(max_num, len(images))]
    features, labels = [None] * len(images), [None] * len(images)
    for i, fname in tqdm(enumerate(images)):
        # tqdm主要作用是用于显示进度
        features[i] = Image.open('%s/JPEGImages/%s.jpg' % (root, fname)).convert("RGB")
        labels[i] = Image.open('%s/SegmentationClass/%s.png' % (root, fname)).convert("RGB")
    return features, labels # PIL image

voc_dir = "F:/PyCharm/Learning_pytorch/data/VOCdevkit/VOC2012"
train_features, train_labels = read_voc_images(voc_dir, max_num=100)


n = 5
imgs = train_features[0:n] + train_labels[0:n]
d2l.show_images(imgs, 2, n)
plt.show()

# 列出标签中每个RGB颜色的值及其标注的类别
# 本函数已保存在d2lzh_pytorch中方便以后使用
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
# 本函数已保存在d2lzh_pytorch中方便以后使用
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

colormap2label = torch.zeros(256 ** 3, dtype=torch.uint8)
for i, colormap in enumerate(VOC_COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

# 本函数已保存在d2lzh_pytorch中方便以后使用
def voc_label_indices(colormap, colormap2label):
    """
    convert colormap (PIL image) to colormap2label (uint8 tensor).
    """
    colormap = np.array(colormap.convert("RGB")).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    '''
    这里需要注意的是，label图片通过查看可以看到，其其实是有边框的，以第一张飞机图片举例
    整个图片颜色大致可分为：背景（黑色）；飞机（红色）；轮廓（白色）
    而通过colormap2label[idx]来索引的时候，之所以没返回轮廓信息，是因为轮廓在最开始的
    colormap2label 标签的构成中就没有设定轮廓的编号，默认为0，同背景一致
    '''
    return colormap2label[idx]


# 以第一张飞机图片为例
y = voc_label_indices(train_labels[0], colormap2label)
print(y[105:115, 130:140], VOC_CLASSES[1])
# 以第四张鸟图片为例
bir = voc_label_indices(train_labels[3],colormap2label)
print(bir[360:380,30:50])
print("*" * 50)

# 预处理数据（随机裁剪）
# 本函数已保存在d2lzh_pytorch中方便以后使用
def voc_rand_crop(feature, label, height, width):
    """
    Random crop feature (PIL image) and label (PIL image).
    """
    i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            feature, output_size=(height, width))

    feature = torchvision.transforms.functional.crop(feature, i, j, h, w)
    label = torchvision.transforms.functional.crop(label, i, j, h, w)

    return feature, label

n = 5
# 裁剪次数设定为5
imgs = []
for _ in range(n):
    imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
d2l.show_images(imgs[::2] + imgs[1::2], 2, n)
plt.show()

# 自定义语义分割数据集类
# 本函数已保存在d2lzh_pytorch中方便以后使用
class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, crop_size, voc_dir, colormap2label, max_num=None):
        """
        crop_size: (h, w)
        """
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.tsf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.rgb_mean,
                                             std=self.rgb_std)
        ])
        # 传入图片的标准化

        self.crop_size = crop_size  # (h, w)
        features, labels = read_voc_images(root=voc_dir,
                                           is_train=is_train,
                                           max_num=max_num)
        self.features = self.filter(features)   # PIL image
        self.labels = self.filter(labels)       # PIL image
        self.colormap2label = colormap2label
        print('read ' + str(len(self.features)) + ' valid examples')

    def filter(self, imgs):
        return [img for img in imgs if (
            img.size[1] >= self.crop_size[0] and
            img.size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        # 从内存中读取特征图和标签图
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)

        return (self.tsf(feature), # float32 tensor
                voc_label_indices(label, self.colormap2label)) # uint8 tensor

    def __len__(self):
        return len(self.features)


crop_size = (320, 480)
max_num = 100
voc_train = VOCSegDataset(True, crop_size, voc_dir, colormap2label, max_num)
voc_test = VOCSegDataset(False, crop_size, voc_dir, colormap2label, max_num)

# 设批量大小为64，分别定义训练集和测试集的迭代器。
batch_size = 64
num_workers = 0 if sys.platform.startswith('win32') else 4
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(voc_test, batch_size, drop_last=True,
                             num_workers=num_workers)

for X, Y in train_iter:
    # X为原始数据，Y为标签
    print(X.dtype, X.shape)
    print(y.dtype, Y.shape)
    break


print("*" * 50)