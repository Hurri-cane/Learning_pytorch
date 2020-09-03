# 本书链接https://tangshusen.me/Dive-into-DL-PyTorch/#/
# 锚框
# 注释：黄文俊
# E-mail：hurri_cane@qq.com

from matplotlib import pyplot as plt
import os
import json
import numpy as np
import torch
import torchvision
from PIL import Image

import sys
sys.path.append("..")
import d2lzh_pytorch as d2l

data_dir = 'F:/PyCharm/Learning_pytorch/data/pikachu'

assert os.path.exists(os.path.join(data_dir, "train"))

# 本类已保存在d2lzh_pytorch包中方便以后使用
class PikachuDetDataset(torch.utils.data.Dataset):
    """皮卡丘检测数据集类"""
    def __init__(self, data_dir, part, image_size=(256, 256)):
        assert part in ["train", "val"]
        self.image_size = image_size
        self.image_dir = os.path.join(data_dir, part, "images")
        # os.path.join作用为路径拼接，功能举例如下：
        '''
        Path1 = 'home'
        Path2 = 'develop'
        Path3 = 'code'
        Path10 = Path1 + Path2 + Path3
        Path20 = os.path.join(Path1,Path2,Path3)
        print ('Path10 = ',Path10)
        print ('Path20 = ',Path20)
        输出为：
        Path10 =  homedevelopcode
        Path20 =  home\develop\code
        '''

        with open(os.path.join(data_dir, part, "label.json")) as f:
            self.label = json.load(f)

        self.transform = torchvision.transforms.Compose([
            # 将 PIL 图片转换成位于[0.0, 1.0]的floatTensor, shape (C x H x W)
            torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        image_path = str(index + 1) + ".png"

        cls = self.label[image_path]["class"]
        label = np.array([cls] + self.label[image_path]["loc"],
                         dtype="float32")[None, :]

        PIL_img = Image.open(os.path.join(self.image_dir, image_path)
                            ).convert('RGB').resize(self.image_size)
        img = self.transform(PIL_img)

        sample = {
            "label": label, # shape: (1, 5) [class, xmin, ymin, xmax, ymax]
            "image": img    # shape: (3, *image_size)
        }

        return sample


# 本函数已保存在d2lzh_pytorch包中方便以后使用
def load_data_pikachu(batch_size, edge_size=256, data_dir = 'F:/PyCharm/Learning_pytorch/data/pikachu'):
    """edge_size：输出图像的宽和高"""
    image_size = (edge_size, edge_size)
    train_dataset = PikachuDetDataset(data_dir, 'train', image_size)
    val_dataset = PikachuDetDataset(data_dir, 'val', image_size)


    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=4)

    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=4)
    return train_iter, val_iter

if __name__ == '__main__':
    # 如果不用字主程序前加上if __name__ == '__main__':会导致线程报错
    # 其实通过阅读上面的load_data_pikachu()函数下的代码可以知道，报错的原因是：
    # 在设定train_iter和val_iter时线程数设定为了4，如果不加入if __name__ == '__main__':
    # 可以将num_workers=4改为等于1即可

    # 试了一下，就算num_workers=4改为等于1还是不行，这就比较奇怪了。
    batch_size, edge_size = 32, 256
    train_iter, _ = load_data_pikachu(batch_size, edge_size, data_dir)
    batch = iter(train_iter).next()
    print(batch["image"].shape, batch["label"].shape)
    print("*" * 50)
    imgs = batch["image"][0:10].permute(0, 2, 3, 1)
    # .permute()表示维度换位，
    # 此案例中便是将之前的维度位置0维，1维，2维，3维换为如下排列0维，2维，3维，1维
    bboxes = batch["label"][0:10, 0, 1:]

    axes = d2l.show_images(imgs, 2, 5).flatten()
    # a = zip(axes, bboxes)
    # b = list(a)
    for ax, bb in zip(axes, bboxes):
        d2l.show_bboxes(ax, [bb * edge_size], colors=['R'])
    plt.show()
print("*" * 50)