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
from main import ConditionalCelebA


def show_cam():
    from PIL import Image
    import torchvision.transforms as transforms

    # model = models.densenet121(num_classes=2)
    # print(model)
    # model.load_state_dict(torch.load('./ckpts/ex4/d2c5_densenet121_best.pt'))
    model = models.resnet18(num_classes=2)
    print(model)
    dataset = ConditionalCelebA(_class=5, train=True)

    # 2403
    data, label = dataset[2410]
    data = data.unsqueeze(dim=0)
    label = label.unsqueeze(dim=0)

    raw = transforms.ToPILImage()(data[0])
    raw.save('./count/img/5raw.png')
    plt.imshow(raw)
    plt.show()

    for i in range(4):
        model.load_state_dict(torch.load(f'./ckpts/ex1/d1c{i}_res18_best.pt'))
        target_layers = [model.layer3[-1]]

        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        grayscale_cam = cam(input_tensor=data, target_category=label)
        for img, gray_cam in zip(data, grayscale_cam):
            visualization = show_cam_on_image(img.numpy().transpose((1, 2, 0)), gray_cam, use_rgb=True)
            img = Image.fromarray(visualization)
            img.save(f'./count/img/{i}.png')
            plt.imshow(img)
            plt.show()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    show_cam()
