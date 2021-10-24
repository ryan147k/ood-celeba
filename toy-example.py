#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/10/18 14:04
# DESCRIPTION:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_dataset(mode, p=1, r=1):
    np.random.seed(2)

    num_examples = 10000

    if mode == 0:
        # origin
        y = np.hstack((np.zeros(num_examples), np.ones(num_examples)))
        x1 = np.hstack((np.random.uniform(-1, 0, num_examples), np.random.uniform(0, 1, num_examples)))
        x2 = np.hstack((np.random.uniform(-1, 1, num_examples), np.random.uniform(-1, 1, num_examples)))
        data = zip(np.stack((x1, x2), axis=-1), y)
        dataset = ToyDataset(data)

    elif mode == 1:
        # train1
        y = np.hstack((np.zeros(num_examples), np.ones(num_examples)))
        x1 = np.hstack((np.random.uniform(-1, 0, num_examples), np.random.uniform(0, 1, num_examples)))

        x2 = []
        for _ in range(num_examples):
            p_ = np.random.uniform()
            if p_ < p:
                x2.append(np.random.uniform(-1, 0))
            else:
                x2.append(np.random.uniform(0, 1))
        for _ in range(num_examples):
            p_ = np.random.uniform()
            if p_ < p:
                x2.append(np.random.uniform(0, 1))
            else:
                x2.append(np.random.uniform(-1, 0))

        x2 = np.array(x2)
        data = zip(np.stack((x1, x2), axis=-1), y)
        dataset = ToyDataset(data)

    if mode == 2:
        # train2
        y = np.hstack((np.zeros(num_examples), np.ones(num_examples)))
        x1 = np.hstack((np.random.uniform(-1, 0, num_examples), np.random.uniform(0, 1, num_examples)))
        x2 = np.random.uniform(-r, r, num_examples * 2)
        data = zip(np.stack((x1, x2), axis=-1), y)
        dataset = ToyDataset(data)

    if mode == 3:
        # train3
        y = np.hstack((np.zeros(num_examples), np.ones(num_examples)))
        x1 = np.hstack((np.random.uniform(-1, 0, num_examples), np.random.uniform(0, 1, num_examples)))

        x2 = []
        for _ in range(num_examples):
            p_ = np.random.uniform()
            if p_ < p:
                x2.append(np.random.uniform(-r, 0))
            else:
                x2.append(np.random.uniform(0, r))
        for _ in range(num_examples):
            p_ = np.random.uniform()
            if p_ < p:
                x2.append(np.random.uniform(0, r))
            else:
                x2.append(np.random.uniform(-r, 0))

        x2 = np.array(x2)
        data = zip(np.stack((x1, x2), axis=-1), y)
        dataset = ToyDataset(data)

    return dataset


class ToyDataset(Dataset):
    def __init__(self, dataset):
        super(ToyDataset, self).__init__()

        self.dataset = list(dataset)

    def __getitem__(self, index):
        data, y = self.dataset[index]
        data = torch.Tensor(data)
        y = torch.Tensor([y])
        return data, y

    def __len__(self):
        return len(self.dataset)


def train(model, dataset):
    model = model.to(device)

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    x, y = next(loader.__iter__())

    x = x.to(device)
    y = y.to(device)


    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-2)
    lr_scheduler = lambda x: 1.0 if x < 50000 else 0.8
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler)

    num_epoch = 100000
    for epoch in range(num_epoch):
        y_hat = model(x)

        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

    return model


def val(model, dataset):
    with torch.no_grad():
        model = model.to(device)

        loader = DataLoader(dataset, batch_size=200)
        x, y = next(loader.__iter__())
        x = x.to(device)
        y = y.to(device)

        y_hat = model(x)

        mask = y_hat.ge(0.5).float()
        num_correct = (mask == y).sum().item()
        acc = num_correct / len(y)
    return acc


if __name__ == '__main__':
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

    origin_dataset = get_dataset(mode=0)

    p_l = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
    r_l = [1, 0.8, 0.6, 0.4, 0.2, 0]

    for mode in range(1, 4):

        acc = []
        for i, (p, r) in enumerate(zip(p_l, r_l)):
            print("------------")

            dataset = get_dataset(mode=mode, p=p, r=r)

            data, label = next(DataLoader(dataset, batch_size=200, shuffle=True).__iter__())
            plt.scatter(data.numpy()[:, 0], data.numpy()[:, 1], cmap='rainbow', c=label.numpy())
            plt.ylim(-1.2, 1.2)
            plt.xlim(-1.2, 1.2)

            model = nn.Sequential(
                nn.Linear(2, 1, bias=True),
                nn.Sigmoid()
            )
            model = train(model, dataset)

            para = list(dict(model.named_parameters()).values())
            w0, w1 = float(para[0].squeeze()[0].item()), float(para[0].squeeze()[1].item())
            b = float(para[1].item())

            x = np.arange(-3, 3)
            y = (-w0 * x - b) / w1

            plt.plot(x, y)
            plt.show()

            acc.append(val(model, origin_dataset))
        print(acc)
        print(np.mean(acc))
        print(np.std(acc))
