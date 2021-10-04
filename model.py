#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/10/4 13:01
# DESCRIPTION:
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(AlexNet, self).__init__()

        # 由于MNIST为28x28， 而最初AlexNet的输入图片是227x227的。所以网络层数和参数需要调节
        self.conv1 = nn.Conv2d(3, 96, kernel_size=5, padding=1)  #AlexCONV1(3,96, k=11,s=4,p=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  #AlexPool1(k=3, s=2)
        self.relu1 = nn.ReLU()

        # self.conv2 = nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)  #AlexCONV2(96, 256,k=5,s=1,p=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)  #AlexPool2(k=3,s=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=1)  #AlexCONV3(256,384,k=3,s=1,p=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1)  #AlexCONV4(384, 384, k=3,s=1,p=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=1)  #AlexCONV5(384, 256, k=3, s=1,p=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  #AlexPool3(k=3,s=2)
        self.relu3 = nn.ReLU()

        self.fc6 = nn.Linear(256*3*3, 1024)  #AlexFC6(256*6*6, 4096)
        self.fc7 = nn.Linear(1024, 512)  #AlexFC6(4096,4096)
        self.fc8 = nn.Linear(512, num_classes)   #AlexFC6(4096,1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = x.view(-1, 256 * 3 * 3)  # Alex: x = x.view(-1, 256*6*6)
        x = self.fc6(x)
        x = torch.relu(x)
        x = self.fc7(x)
        x = torch.relu(x)
        x = self.fc8(x)
        return x


class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()

        self.c1 = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(3, 6, kernel_size=(3, 3))),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c1(img)
        return output


class C2(nn.Module):
    def __init__(self):
        super(C2, self).__init__()

        self.c2 = nn.Sequential(OrderedDict([
            ('c2', nn.Conv2d(6, 16, kernel_size=(3, 3))),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        ]))

    def forward(self, img):
        output = self.c2(img)
        return output


class C3(nn.Module):
    def __init__(self):
        super(C3, self).__init__()

        self.c3 = nn.Sequential(OrderedDict([
            ('c3', nn.Conv2d(16, 120, kernel_size=(3, 3))),
            ('relu3', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.c3(img)
        return output


class F4(nn.Module):
    def __init__(self):
        super(F4, self).__init__()

        self.f4 = nn.Sequential(OrderedDict([
            ('f4', nn.Linear(1080, 84)),
            ('relu4', nn.ReLU())
        ]))

    def forward(self, img):
        output = self.f4(img)
        return output


class F5(nn.Module):
    def __init__(self, num_classes):
        super(F5, self).__init__()

        self.f5 = nn.Sequential(OrderedDict([
            ('f5', nn.Linear(84, num_classes)),
        ]))

    def forward(self, img):
        output = self.f5(img)
        return output


class LeNet5(nn.Module):
    """
    Input - 1x32x32
    Output - 10
    """
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()

        self.c1 = C1()
        self.c2_1 = C2()
        self.c2_2 = C2()
        self.c3 = C3()
        self.f4 = F4()
        self.f5 = F5(num_classes)

    def forward(self, img):
        output = self.c1(img)

        x = self.c2_1(output)
        output = self.c2_2(output)

        output += x

        output = self.c3(output)
        output = output.view(img.size(0), -1)
        output = self.f4(output)
        output = self.f5(output)
        return output