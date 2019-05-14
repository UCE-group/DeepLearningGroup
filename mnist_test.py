import torchvision
import torch
from torch import nn, optim
import numpy as np
import cv2
import time
from torch.utils.data import DataLoader
import os

transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
     torchvision.transforms.Normalize([0.5], [0.5])])
def get_files(path):
    list = os.listdir(path)
    paths = [path + name for name in list]
    labels = [int(name.split('.')[0]) for name in list]
    return paths, labels

class my_mnist(nn.Module):
    def __init__(self):
        super(my_mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Linear(32*7*7, 10)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        out = self.layer3(x)
        return out

#net.load_state_dict(torch.load('pkl/499net.pth'))
net = torch.load('pkl/499net.pth').cpu()
path = 'test/'

def threshold_filter(threshold, image):
    for i in range(28):
        for j in range(28):
            image[i][j] = 255 - image[i][j]
            if(image[i][j] < threshold):
                image[i][j] = 0
            else:
                image[i][j] = 255
    return image
paths, labels = get_files(path)
threshold = 99 #80-141
for e in range(len(labels)):
    image = cv2.imread(paths[e])   #读取图片
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #变为灰度图
    image = cv2.resize(img_gray, (28,28))   #改变大小
    image = threshold_filter(threshold, image)    #阈值过滤
    image = transform(image)    #转换为Tensor，对像素点进行处理变为[-1,1]区间
    image = image.reshape(-1,1,28,28)

    out = net(image)  #模型处理
    pred = torch.max(out, 1)[1] #得出结果
    print(pred.item(), labels[e])
