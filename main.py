#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/10/4 12:59
# DESCRIPTION:
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchvision as tv
import torchvision.transforms as transforms
from model import *
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import random
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=True)
parser.add_argument('--ex_num', type=str)
parser.add_argument('--dataset_id', type=int)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epoch_num', type=int, default=20)
parser.add_argument('--print_iter', type=int, default=20)
args = parser.parse_args()


class RawCelebA(Dataset):
    def __init__(self, file_list, root='../dataset/CelebA/Img/img_align_celeba'):
        super(RawCelebA, self).__init__()
        self.root = root
        self.file_list = file_list
        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        file_name, label = self.file_list[index]
        img_path = os.path.join(self.root, file_name)
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            data = self.transform(img)
            return data, torch.LongTensor([int(label)])

    def __len__(self):
        return len(self.file_list)


class ClusteredCelebA(RawCelebA):
    def __init__(self, _class, root='./dataset/celeba_clustered'):
        assert 0 <= _class < 5
        file_list = open(os.path.join(root, f'{str(_class)}.txt')).readlines()
        file_list = [_.split() for _ in file_list]

        super(ClusteredCelebA, self).__init__(file_list)


def train(model,
          save_path: str,
          ex_name: str,
          train_dataset,
          val_dataset,
          test_datasets=None):
    # data

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=5)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=5)
    if test_datasets is None:
        test_loaders = []
    else:
        test_loaders = [DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=2)
                        for dataset in test_datasets]

    # model

    # 多GPU运行
    model = nn.DataParallel(model)
    model = model.to(device)
    print(model.module)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    lr_scheduler = lambda x: 1.0 if x < 30 else 0.8
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler)

    # train

    best_val_acc, best_val_iter = 0.0, None  # 记录全局最优信息
    save_model = False

    writer = SummaryWriter()
    iter = 0
    for epoch in range(args.epoch_num):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).squeeze()

            # forward
            y_hat = model(batch_x)
            loss = loss_fn(y_hat, batch_y)
            # backward
            optimizer.zero_grad()
            loss.backward()
            # update
            optimizer.step()

            # 计算精度
            _, pred = y_hat.max(1)
            num_correct = (pred == batch_y).sum().item()
            acc = num_correct / len(batch_y)

            iter += 1
            if iter % args.print_iter == 0:
                # 打印信息
                train_loss, train_acc = loss.item(), acc
                val_loss, val_acc = val(model, val_loader)
                test_info_list = [val(model, loader) for loader in test_loaders]
                print("\n[INFO] Epoch {} Iter {}:\n \
                            \tTrain: Loss {:.4f}, Accuracy {:.4f}\n \
                            \tVal:   Loss {:.4f}, Accuracy {:.4f}".format(epoch + 1, iter,
                                                                          train_loss, train_acc,
                                                                          val_loss, val_acc))
                for ii, (test_loss, test_acc) in enumerate(test_info_list):
                    print("\tTest{}: Loss {:.4f}, Accuracy {:.4f}".format(ii, test_loss, test_acc))

                tensorboard_write(writer=writer,
                                  ex_name=ex_name,
                                  mode_name='{} {}'.format(save_path.split('/')[-1], 'acc'),
                                  value_dict={'train_acc': train_acc,
                                              'val_acc': val_acc},
                                  x_axis=iter)
                tensorboard_write(writer=writer,
                                  ex_name=ex_name,
                                  mode_name='{} {}'.format(save_path.split('/')[-1], 'loss'),
                                  value_dict={'train_loss': train_loss,
                                              'val_loss': val_loss},
                                  x_axis=iter)

                # 更新全局最优信息
                if val_acc > best_val_acc:
                    best_val_acc, best_val_iter = val_acc, iter
                    save_model = True
                if save_model:
                    # 保存模型
                    torch.save(model.module.state_dict(), '{}_best.pt'.format(save_path))
                    save_model = False

                print("\t best val   acc so far: {:.4} Iter: {}".format(best_val_acc, best_val_iter))

                # torch.save(model.module.state_dict(), '{}_i{}.pt'.format(save_path, iter))

        # 保存模型
        # torch.save(model.module.state_dict(), '{}_e{}.pt'.format(save_path, epoch))
        scheduler.step()


@torch.no_grad()
def val(model, dataloader):
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    loss_sum = 0
    acc_sum = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).squeeze()

        # forward
        y_hat = model(batch_x)
        loss = loss_fn(y_hat, batch_y)

        loss_sum += loss.item()
        # 计算精度
        _, pred = y_hat.max(1)
        num_correct = (pred == batch_y).sum().item()
        acc = num_correct / len(batch_y)
        acc_sum += acc

    model.train()

    return loss_sum / len(dataloader), acc_sum / len(dataloader)


def plot(img):
    # (c, h, w) -> (h, w, c)
    img = np.transpose(img, (1, 2, 0))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def tensorboard_write(writer, ex_name: str, mode_name, value_dict, x_axis):
    """
    tensorboardX 作图
    :param writer:
    :param ex_name: 实验名称
    :param mode_name: 模型名称+数据 eg. resnet18 acc
    :param value_dict:
    :param x_axis:
    :return:
    """
    writer.add_scalars(main_tag='{}/{}'.format(ex_name, mode_name),
                       tag_scalar_dict=value_dict,
                       global_step=x_axis)


class Experiment:
    """
    记录每一次的实验设置
    """
    num_cluster = 5

    @staticmethod
    def _mkdir(save_dir):
        """
        如果目录不存在, 则创建
        :param save_dir: 模型检查点保存目录
        :return:
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    @staticmethod
    def _get_loader(dataset):
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    @classmethod
    def _split_train_val(cls, dataset):
        """
        将dataset分成两个dataset: train & val
        :param dataset:
        :return:
        """
        class _Dataset(Dataset):
            def __init__(self, dataset, _train=True):
                self.dataset = dataset

                index_list = list(range(len(dataset)))
                random.seed(2)
                random.shuffle(index_list)

                split_num = int(len(dataset) * 9 / 10)
                train_index_list = index_list[:split_num]
                val_index_list = index_list[split_num:]

                self.index_list = train_index_list if _train else val_index_list

            def __getitem__(self, index):
                data, label = self.dataset[self.index_list[index]]
                return data, label

            def __len__(self):
                return len(self.index_list)

            def collate_fn(self, batch):
                return self.dataset.collate_fn(batch)

        return _Dataset(dataset, _train=True), _Dataset(dataset, _train=False)

    @classmethod
    def _basic_test(cls, model, dataset):
        """
        获取模型在某个测试集上的loss和acc
        :param model:
        :param dataset:
        :return:
        """
        model = nn.parallel.DataParallel(model)
        model.to(device)

        loader = cls._get_loader(dataset)
        loss, acc = val(model, loader)
        # print('loss {} acc {}'.format(loss, acc))
        return loss, acc

    @classmethod
    def _test(cls, model, save_dir, model_name, datasets, ckpt_nums):
        """
        获取一系列模型检查点在测试集上的准确率
        :param model: 待测试模型
        :param save_dir: 模型检查点保存目录
        :param model_name: 模型检查点名称
        :param datasets: 测试集列表
        :param ckpt_nums: 检查点iter (同时是x轴坐标)
        :return:
        """
        acc_tests = [[] for _ in datasets]
        # 记录每个epoch的模型在数据集上的准确率
        for num in tqdm(ckpt_nums):
            model.load_state_dict(torch.load(f'{save_dir}/{model_name}_e{num}.pt'))
            # print('[INFO] Iter {}'.format(num), end='\n\t')
            for i, dataset in enumerate(datasets):
                _, acc = cls._basic_test(model, dataset)
                acc_tests[i].append(acc)
        return acc_tests

    @staticmethod
    def _plot(scalars_list, labels, x_axis):
        """
        画曲线图
        :param scalars_list: List[List], 每一个子List就是一条曲线
        :param labels: 每一个子List所代表的标签
        :param x_axis: x轴数值
        :return:
        """
        for i in range(len(scalars_list)):
            plt.plot(x_axis, scalars_list[i], label=labels[i])
            plt.xlabel('Iter')
            plt.ylabel('Acc')
            plt.legend()
            plt.show()

    @classmethod
    def _plot_confusion_matrix(cls, model, data_dir, dataset_name):
        """混淆矩阵可视化"""
        model = nn.parallel.DataParallel(model)
        model.to(device)

        # dataset = ColorMNIST(data_dir=data_dir, dataset_name=dataset_name)
        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        # plot_confusion_matrix(get_confusion_matrix(model, loader, device))

    @classmethod
    def _ex(cls, ex_name, _train, model, model_name, save_dir, train_dataset, val_dataset, test_dataset=None):

        if _train:
            train(model, save_path='{}/{}'.format(save_dir, model_name),
                  ex_name=ex_name, train_dataset=train_dataset,
                  val_dataset=val_dataset, test_datasets=test_dataset)
        else:
            model.load_state_dict(torch.load(f'{save_dir}/{model_name}_best.pt'))
            acc_list = []
            _, acc = cls._basic_test(model, val_dataset)
            acc_list.append(acc)

            for i in range(1, cls.num_cluster):
                if args.dataset_id == 0:
                    dataset = ClusteredCelebA(i)
                # elif args.dataset_id == 1:
                #     dataset = CorrelationMNIST(i)
                # elif args.dataset_id == 2:
                #     dataset = DiversityMNIST(i)

                _, acc = cls._basic_test(model, dataset)
                acc_list.append(acc)

            plt.plot(range(cls.num_cluster), acc_list, '-o')
            plt.title(f'{model_name}')
            plt.show()

    @classmethod
    def ex1(cls, _train=False):
        args.batch_size = 256
        args.epoch_num = 50
        print(args)

        ex_name = 'ex1'
        save_dir = './ckpts/ex1/1004'
        cls._mkdir(save_dir)

        model = tv.models.resnet18(num_classes=2)
        model_name = 'res18'

        if args.dataset_id == 0:
            dataset = ClusteredCelebA(0)
        elif args.dataset_id == 1:
            dataset = ClusteredCelebA(0)
        elif args.dataset_id == 2:
            dataset = ClusteredCelebA(0)
        model_name = f'd{str(args.dataset_id)}_{model_name}'

        train_dataset, val_dataset = cls._split_train_val(dataset)
        cls._ex(ex_name, _train, model, model_name, save_dir, train_dataset, val_dataset)

    @classmethod
    def ex2(cls, _train=False):
        args.batch_size = 512
        args.epoch_num = 50
        print(args)

        ex_name = 'ex2'
        save_dir = './ckpts/ex2/1004'
        cls._mkdir(save_dir)

        model = tv.models.AlexNet(num_classes=2)
        model_name = 'alexnet'

        if args.dataset_id == 0:
            dataset = ClusteredCelebA(0)
        elif args.dataset_id == 1:
            dataset = ClusteredCelebA(0)
        elif args.dataset_id == 2:
            dataset = ClusteredCelebA(0)
        model_name = f'd{str(args.dataset_id)}_{model_name}'

        train_dataset, val_dataset = cls._split_train_val(dataset)
        cls._ex(ex_name, _train, model, model_name, save_dir, train_dataset, val_dataset)


if __name__ == '__main__':
    _train = True
    if args.debug is True:
        args.ex_num = '1'
        args.dataset_id = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = '7'
        _train = False

    ex = getattr(Experiment, f'ex{args.ex_num.strip()}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ex(_train)
