#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# AUTHOR: Ryan Hu
# DATE: 2021/10/4 12:59
# DESCRIPTION: dataset_type represent different dataset (e.g. celeba_diversity)
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import os
from tensorboardX import SummaryWriter
import random
from PIL import Image
import csv


parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=True)
parser.add_argument('--ex_num', type=str)
parser.add_argument('--dataset_type', type=int)
parser.add_argument('--train_class', type=int)
parser.add_argument('--test_classes', type=list)

parser.add_argument('--num_workers', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epoch_num', type=int, default=100)
parser.add_argument('--print_iter', type=int, default=10)
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


class ShiftedCelebA(RawCelebA):
    def __init__(self, _class, root='./dataset/celeba_shift'):
        assert 0 <= _class < 7
        file_list = open(os.path.join(root, f'{str(_class)}.txt')).readlines()
        file_list = [_.split() for _ in file_list]

        super(ShiftedCelebA, self).__init__(file_list)


class ClusteredCelebA(RawCelebA):
    def __init__(self, _class, root='./dataset/celeba_clustered'):
        assert 0 <= _class < 5
        file_list = open(os.path.join(root, f'{str(_class)}.txt')).readlines()
        file_list = [_.split() for _ in file_list]

        super(ClusteredCelebA, self).__init__(file_list)


class ConditionalCelebA(RawCelebA):
    def __init__(self, _class, root='./dataset/celeba_correlation', train=True):
        assert 0 <= _class < 6

        file = 'train' if train else 'test'
        file_list = open(os.path.join(root, f'{str(_class)}/{file}.txt')).readlines()
        file_list = [_.split() for _ in file_list]

        super(ConditionalCelebA, self).__init__(file_list)


class MarginalCelebA(RawCelebA):
    def __init__(self, _class, root='./dataset/celeba_diversity', train=True):
        assert 0 <= _class < 4

        file = 'train' if train else 'test'
        file_list = open(os.path.join(root, f'{str(_class)}/{file}.txt')).readlines()
        file_list = [_.split() for _ in file_list]

        super(MarginalCelebA, self).__init__(file_list)


def train(model,
          save_dir: str,
          model_name: str,
          ex_name: str,
          train_dataset,
          val_dataset,
          test_datasets=None):
    # data

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
    if test_datasets is None:
        test_loaders = []
    else:
        test_loaders = [DataLoader(dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)
                        for dataset in test_datasets]

    # model

    # 多GPU运行
    model = nn.DataParallel(model)
    model = model.to(device)
    print(model.module)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    lr_scheduler = lambda x: 1.0 if x < 30 else 0.8
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_scheduler)

    # train

    best_val_acc, best_val_iter = 0.0, None  # 记录全局最优信息
    save_model = False

    writer = SummaryWriter('./runs/{}'.format(ex_name))
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
                accs = []
                # 打印信息
                train_loss, train_acc = loss.item(), acc
                val_loss, val_acc = val(model, val_loader)
                test_info_list = [val(model, loader) for loader in test_loaders]
                print("\n[INFO] Epoch {} Iter {}:".format(epoch, iter))
                print("\t\t\t\t\t\tTrain: Loss {:.4f}, Accuracy {:.4f}".format(train_loss, train_acc))
                print("\t\t\t\t\t\tVal:   Loss {:.4f}, Accuracy {:.4f}".format(val_loss, val_acc))
                accs.append(val_acc)

                test_acc_dict, test_loss_dict = {}, {}
                for ii, (test_loss, test_acc) in enumerate(test_info_list):
                    print("\t\t\t\t\t\tTest{}: Loss {:.4f}, Accuracy {:.4f}".format(ii, test_loss, test_acc))
                    accs.append(test_acc)
                    test_acc_dict[f'test{ii}_acc'] = test_acc
                    test_loss_dict[f'test{ii}_loss'] = test_loss
                csv.writer(open('./count/tmp.csv', 'a')).writerow(accs)

                acc_value_dict = {'train_acc': train_acc,
                                  'val_acc': val_acc}
                loss_value_dict = {'train_loss': train_loss,
                                   'val_loss': val_loss}
                acc_value_dict.update(
                    test_acc_dict
                )
                loss_value_dict.update(
                    test_loss_dict
                )
                tensorboard_write(writer=writer,
                                  ex_name=ex_name,
                                  mode_name='{} {}'.format(model_name, 'acc'),
                                  value_dict=acc_value_dict,
                                  x_axis=iter)
                tensorboard_write(writer=writer,
                                  ex_name=ex_name,
                                  mode_name='{} {}'.format(model_name, 'loss'),
                                  value_dict=loss_value_dict,
                                  x_axis=iter)

                # 更新全局最优信息
                if val_acc > best_val_acc:
                    best_val_acc, best_val_iter = val_acc, iter
                    save_model = True
                if save_model:
                    # 保存模型
                    torch.save(model.module.state_dict(), os.path.join(save_dir, f'{model_name}_best.pt'))
                    save_model = False

                print("\t best val   acc so far: {:.4} Iter: {}".format(best_val_acc, best_val_iter))

        scheduler.step()

        # 保存模型
        save_epoch = False
        if save_epoch:
            epoch_save_path = os.path.join(save_dir, 'iter')
            if not os.path.exists(epoch_save_path):
                os.mkdir(epoch_save_path)
            torch.save(model.module.state_dict(), f'{epoch_save_path}/{model_name}_e{epoch}.pt')


@torch.no_grad()
def val(model, dataloader):
    """
    batch级别的loss & 样本级别的acc
    :param model:
    :param dataloader:
    :return:
    """
    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    loss_sum = 0
    correct_sum = 0
    num_x = 0
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).squeeze()

        # forward
        y_hat = model(batch_x)
        loss = loss_fn(y_hat, batch_y)

        # 计算精度
        _, pred = y_hat.max(1)
        num_correct = (pred == batch_y).sum().item()

        num_x += len(batch_x)
        loss_sum += loss.item()
        correct_sum += num_correct

    model.train()

    return loss_sum / len(dataloader), correct_sum / num_x


def tensorboard_write(writer, ex_name, mode_name, value_dict, x_axis):
    """
    tensorboardX 作图
    :param writer:
    :param ex_name:
    :param mode_name: 模型名称+数据 eg. resnet18 acc
    :param value_dict:
    :param x_axis:
    :return:
    """
    writer.add_scalars(main_tag='{}/{}'.format(ex_name, mode_name),
                       tag_scalar_dict=value_dict,
                       global_step=x_axis)


def split_train_val(dataset, split_num=None):
    """
    将dataset分成两个dataset: train & val
    :param split_num:
    :param dataset:
    :return:
    """

    class _Dataset(Dataset):
        def __init__(self, dataset, _train, split_num):
            self.dataset = dataset

            index_list = list(range(len(dataset)))
            random.seed(2)
            random.shuffle(index_list)

            if split_num is None:
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

    return _Dataset(dataset, _train=True, split_num=split_num), \
        _Dataset(dataset, _train=False, split_num=split_num)


class Experiment:
    """
    记录每一次的实验设置
    """

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
    def _get_dataset(dataset_type, _class, train=True):
        """
        根据 args.dataset_type 来获取 dataset
        :param dataset_type:
        :param _class:
        :return:
        """
        assert 0 <= dataset_type < 4

        _class = int(_class)

        if dataset_type == 0:
            return ShiftedCelebA(_class=_class)
        elif dataset_type == 1:
            return MarginalCelebA(_class=_class, train=train)
        elif dataset_type == 2:
            return ConditionalCelebA(_class=_class, train=train)
        else:
            return ClusteredCelebA(_class=_class)

    @classmethod
    def _ex(cls, model, save_dir, model_name, ex_name, dataset_type, train_class, test_classes):
        """
        模型训练 or 测试
        :param model:
        :param save_dir: 模型检查点保存目录
        :param model_name: 模型名称
        :param ex_name:
        :return:
        """
        model_name = f'd{str(dataset_type)}c{str(train_class)}_{model_name}'

        dataset = cls._get_dataset(dataset_type, _class=train_class)

        train_dataset, val_dataset = split_train_val(dataset)
        if test_classes is None:
            test_classes = []
        test_datasets = [cls._get_dataset(dataset_type, _class=test_class) for test_class in test_classes]

        train(model, save_dir=save_dir, model_name=model_name, ex_name=ex_name,
              train_dataset=train_dataset, val_dataset=val_dataset, test_datasets=test_datasets)

    @classmethod
    def ex1(cls):
        # args.batch_size = 256
        args.batch_size = 16
        args.batch_size = int(args.batch_size / 1)
        args.num_epoch = 10
        print(args)

        ex_name = 'ex1'
        save_dir = './ckpts/ex1'
        cls._mkdir(save_dir)

        model = models.resnet18(num_classes=2)
        model_name = 'res18'

        return cls._ex(model, save_dir, model_name, ex_name,
                       args.dataset_type, args.train_class, args.test_classes)

    @classmethod
    def ex2(cls):
        args.batch_size = 512
        args.batch_size = int(args.batch_size / 1)
        print(args)

        ex_name = 'ex2'
        save_dir = './ckpts/ex2'
        cls._mkdir(save_dir)

        model = models.AlexNet(num_classes=2)
        model_name = 'alexnet'

        return cls._ex(model, save_dir, model_name, ex_name,
                       args.dataset_type, args.train_class, args.test_classes)

    @classmethod
    def ex3(cls):
        args.batch_size = 128
        args.batch_size = int(args.batch_size / 1)
        print(args)

        ex_name = 'ex3'
        save_dir = './ckpts/ex3'
        cls._mkdir(save_dir)

        model = models.vgg11(num_classes=2)
        model_name = 'vgg11'

        return cls._ex(model, save_dir, model_name, ex_name,
                       args.dataset_type, args.train_class, args.test_classes)

    @classmethod
    def ex4(cls):
        args.batch_size = 64
        args.batch_size = int(args.batch_size / 1)
        print(args)

        ex_name = 'ex4'
        save_dir = './ckpts/ex4'
        cls._mkdir(save_dir)

        model = models.DenseNet(num_classes=2)
        model_name = 'densenet121'

        return cls._ex(model, save_dir, model_name, ex_name,
                       args.dataset_type, args.train_class, args.test_classes)

    @classmethod
    def ex5(cls):
        args.batch_size = 256
        args.batch_size = int(args.batch_size / 1)
        args.epoch_num = 200
        print(args)

        ex_name = 'ex5'
        save_dir = './ckpts/ex5'
        cls._mkdir(save_dir)

        model = models.squeezenet1_0(num_classes=2)
        model_name = 'squeezenet1_0'

        return cls._ex(model, save_dir, model_name, ex_name,
                       args.dataset_type, args.train_class, args.test_classes)

    @classmethod
    def ex6(cls):
        args.batch_size = 64
        args.batch_size = int(args.batch_size / 1)
        print(args)

        ex_name = 'ex6'
        save_dir = './ckpts/ex6'
        cls._mkdir(save_dir)

        model = models.resnext50_32x4d(num_classes=2)
        model_name = 'resnext50_32x4d'

        return cls._ex(model, save_dir, model_name, ex_name,
                       args.dataset_type, args.train_class, args.test_classes)


def tst():
    def _basic_test(model, dataset, batch_size=args.batch_size):
        """
        获取模型在某个测试集上的loss和acc
        :param batch_size:
        :param model:
        :param dataset:
        :return:
        """
        model = nn.parallel.DataParallel(model)
        model.to(device)

        loader = DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers)
        loss, acc = val(model, loader)
        return loss, acc

    def t():
        model_list = [models.resnet18(num_classes=2),
                      models.alexnet(num_classes=2),
                      models.vgg11(num_classes=2),
                      models.densenet121(num_classes=2),
                      models.squeezenet1_0(num_classes=2),
                      models.resnext50_32x4d(num_classes=2)]

        res = []

        dataset_type = 1
        for i in range(4):
            res.append([])

            ckpt_list = [f'./ckpts/ex1/d{dataset_type}c{i}_res18_best.pt',
                         f'./ckpts/ex2/d{dataset_type}c{i}_alexnet_best.pt',
                         f'./ckpts/ex3/d{dataset_type}c{i}_vgg11_best.pt',
                         f'./ckpts/ex4/d{dataset_type}c{i}_densenet121_best.pt',
                         f'./ckpts/ex5/d{dataset_type}c{i}_squeezenet1_0_best.pt',
                         f'./ckpts/ex6/d{dataset_type}c{i}_resnext50_32x4d_best.pt']

            dataset_ood = MarginalCelebA(_class=i, train=False)

            for model, ckpt in zip(model_list, ckpt_list):
                model.load_state_dict(torch.load(ckpt))
                _, ood = _basic_test(model, dataset_ood, batch_size=64)
                res[-1].append(ood * 100)

        for l in res:
            for n in l:
                print(n, end=', ')
            print()

    global device
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t()


if __name__ == '__main__':
    if args.debug is not True:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ex = getattr(Experiment, f'ex{args.ex_num.strip()}')
        ex()
    else:
        tst()
