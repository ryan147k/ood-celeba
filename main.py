# #!/usr/bin/env python
# # -*- coding:UTF-8 -*-
# # AUTHOR: Ryan Hu
# # DATE: 2021/9/29 20:27
# # DESCRIPTION:
from multiprocessing import Manager, Pool
import linecache
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import random
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans


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


class ClusteredCelebA:
    file_list = [_.split() for _ in open('./count/img4cluster.txt').readlines()]

    celeba = RawCelebA(file_list)
    loader = DataLoader(celeba, batch_size=len(celeba), num_workers=20)
    encoder = tv.models.resnet50(pretrained=True)

    num_class = 2

    @classmethod
    def t(cls):
        print('a')
        a = cls._get_coding(0)
        pass

    @classmethod
    def _get_data_index_list(cls, data_class):
        """
        获取某个数字的index列表
        :param data_class:  取哪个类别数据的index
        :param num_class: 类别数量
        :return:
        """
        _, label = next(cls.loader.__iter__())
        label = label.squeeze()
        index_lists = [[] for _ in range(cls.num_class)]
        for idx, label in enumerate(label.tolist()):
            index_lists[label].append(idx)
        return index_lists[data_class]

    @classmethod
    def _get_coding(cls, data_class):
        def _hook(module, input, output):
            coding.append(input[0].cpu().detach())

        index_list = cls._get_data_index_list(data_class)

        data, label = next(cls.loader.__iter__())
        label = label.squeeze()
        data, label = data[index_list], label[index_list]  # 取出某个数字的数据

        coding = []  # 用于接收钩子函数的输出
        cls.encoder.fc.register_forward_hook(_hook)
        cls.encoder(data)  # TODO: batch_size太大, 直接放内存放不下

        return coding[0]

    @classmethod
    def coding2pkl(cls):
        coding_list = []
        for data_class in range(cls.num_class):
            coding = cls._get_coding(data_class)
            coding_list.append(coding)
        pickle.dump(coding_list, open('./count/coding_list.pkl', 'wb'))

    @classmethod
    def kmeans2pkl(cls):
        coding_list = pickle.load(open('./count/coding_list.pkl', 'rb'))
        cluster_list = []
        for data_class in tqdm(range(cls.num_class)):
            coding = coding_list[data_class]
            cluster = KMeans(n_clusters=5, random_state=2).fit(coding)
            cluster_list.append(cluster)
        pickle.dump(cluster_list, open('./count/cluster_list.pkl', 'wb'))

    @classmethod
    def _get_cluster_index_list(cls, data_class):
        """
        获取某个类别聚类后的index列表
        :param _class:
        :return:
        """
        cluster_list = pickle.load(open('./count/cluster_list.pkl', 'rb'))
        cluster = cluster_list[data_class]

        index_list = [[] for _ in range(5)]
        for i, v in enumerate(cluster.labels_):
            index_list[v].append(i)
        return index_list

    @classmethod
    def _get_ni_info(cls, data_class):
        def _ni_index(cluster_0, cluster_1):
            cluster_0 = cluster_0.numpy()
            cluster_1 = cluster_1.numpy()
            mean_0 = np.mean(cluster_0, axis=0)
            mean_1 = np.mean(cluster_1, axis=0)
            std = np.std(np.concatenate((cluster_0, cluster_1), axis=0))
            z = (mean_0 - mean_1) / std
            ni = np.linalg.norm(z)
            return ni

        coding_list = pickle.load(open('./count/coding_list.pkl', 'rb'))
        coding = coding_list[data_class]

        cluster_index_list = cls._get_cluster_index_list(data_class)

        coding_basic_cluster = coding[cluster_index_list[0]]

        ni_list = []
        for cluster_index in cluster_index_list:
            coding_compared_cluster = coding[cluster_index]
            ni = _ni_index(coding_basic_cluster, coding_compared_cluster)
            ni_list.append(ni)

        ni_rank = sorted(range(len(ni_list)), key=lambda k: ni_list[k])
        ni_list = sorted(ni_list)
        return ni_list, ni_rank

    @classmethod
    def print_ni_info(cls):
        ni_info_list = []
        for data_class in range(10):
            ni_info = cls._get_ni_info(data_class)
            print(ni_info)
            ni_info_list.append(ni_info)

        ni, _ = zip(*ni_info_list)
        ni = np.array(ni)
        print(np.mean(ni, axis=0).tolist())  # [ 0.0, 9.22905, 11.084345, 12.117267, 14.091411]


class CelebA:
    root = '../dataset/CelebA/Anno/list_attr_celeba.txt'

    @classmethod
    def _run(cls, i, res, args):
        line = linecache.getline(cls.root, i)
        info = line.split()

        flag = True
        for attr_id, attr_mode in args:
            if int(info[attr_id]) != attr_mode:
                flag = False
                break
        if flag:
            res.append(info[0])

    @classmethod
    def _get_attrs_list(cls, attr_id_list=(), attr_mode_list=()):
        """
        获取lines里拥有某些属性的图片列表
        :param attr_id_list: 属性id列表
        :param attr_mode_list: 属性模式列表 1 or -1
        :return:
        """
        lines = list(range(3, 202602))

        manager = Manager()
        res = manager.list()
        args = list(zip(attr_id_list, attr_mode_list))

        pool = Pool(processes=20)
        pool.starmap(cls._run,
                     zip(lines, [res] * len(lines), [args] * len(lines)))
        pool.close()
        pool.join()
        return list(res)

    @classmethod
    def print_hair_info_with_gender(cls):
        with open(cls.root, 'r') as f:
            f.readline()
            attr_list = f.readline().split()

        male = {}
        for attr_id in [9, 10, 12, 18]:
            attr_name = attr_list[attr_id - 1]
            res = cls._get_attrs_list(attr_id_list=(21, attr_id), attr_mode_list=(1, 1))
            male[attr_name] = len(res)
        female = {}
        for attr_id in [9, 10, 12, 18]:
            attr_name = attr_list[attr_id - 1]
            res = cls._get_attrs_list(attr_id_list=[21, attr_id], attr_mode_list=[-1, 1])
            female[attr_name] = len(res)
        print(male)
        print(female)

    @classmethod
    def celeba2txt(cls):
        random.seed(2)
        # 聚类
        with open('./count/img4cluster.txt', 'w') as f:
            for label in [0, 1]:
                gender_id = 21
                gender = -1 if label == 0 else 1

                img_list = cls._get_attrs_list((gender_id,), (gender,))
                img_list = random.sample(img_list, 20000)

                img_list = [f'{_} {label}\n' for _ in img_list]
                f.writelines(img_list)


ClusteredCelebA.coding2pkl()
