#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/11/3 11:35
# DESCRIPTION:
import torch
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.models as models
import matplotlib.pyplot as plt
import os
from main import ShiftedCelebA


def show_cam():
    from PIL import Image
    model = models.resnet18(num_classes=2)
    print(model)
    model.load_state_dict(torch.load('./ckpts/ex1/d0c2_res18_best.pt'))
    target_layers = [model.layer3[-1]]

    loader = DataLoader(ShiftedCelebA(_class=5), batch_size=10, shuffle=True, num_workers=3)
    data, label = next(loader.__iter__())
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    grayscale_cam = cam(input_tensor=data, target_category=label)
    for img, gray_cam in zip(data, grayscale_cam):
        visualization = show_cam_on_image(img.numpy().transpose((1, 2, 0)), gray_cam, use_rgb=True)
        img = Image.fromarray(visualization)
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    show_cam()
