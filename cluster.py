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
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.pyplot as plt


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
    file_list = [_.split() for _ in open('./dataset/celeba_shift/0.txt').readlines()]
    encoder = tv.models.resnet50(pretrained=True)

    num_class = 2
    num_cluster = 4

    @classmethod
    def t(cls):
        """
        For test
        :return:
        """
        a = cls._get_ni_info(0)
        print(a)

    @classmethod
    def _get_img_list(cls, data_class):
        """
        根据类别获取img列表
        :param data_class:
        :return:
        """
        img_list = []
        for img, label in cls.file_list:
            if int(label) == data_class:
                img_list.append((img, label))
        return img_list

    @classmethod
    def _get_coding(cls, data_class):
        def _hook(module, input, output):
            hook_res.append(input[0].cpu().detach())

        def _forward(model, dataset):
            with torch.no_grad():
                device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                loader = DataLoader(dataset, batch_size=128, num_workers=10)
                for data, _ in loader:
                    data = data.to(device)
                    model(data)  # TODO: batch_size太大, 直接放内存放不下

        img_list = cls._get_img_list(data_class)

        hook_res = []  # 用于接收钩子函数的输出
        cls.encoder.fc.register_forward_hook(_hook)

        celeba = RawCelebA(img_list)
        _forward(cls.encoder, celeba)

        coding = torch.cat(hook_res, dim=0)

        return coding

    @classmethod
    def plot_pca_coding(cls):
        coding_list = []
        label_list = []
        for i in range(cls.num_class):
            coding = cls._get_coding(i)
            label = torch.zeros(len(coding)) + i
            coding_list.append(coding)
            label_list.append(label)
        coding = torch.cat(coding_list, dim=0)
        label = torch.cat(label_list, dim=0)

        coding = PCA(n_components=2, random_state=2).fit_transform(coding)

        index_list = range(len(coding))
        random.seed(2)
        index_list = random.sample(index_list, 1000)
        coding = coding[index_list]
        label = label[index_list]

        label = label.numpy().tolist()
        plt.scatter(coding[:, 0], coding[:, 1], c=label, cmap='rainbow')
        plt.show()

    @classmethod
    def coding2pkl(cls):
        coding_list = []
        for data_class in range(cls.num_class):
            coding = cls._get_coding(data_class)
            coding_list.append(coding)
        pickle.dump(coding_list, open('./count/coding_list.pkl', 'wb'))

    @classmethod
    def kmeans2pkl(cls, random_state=2):
        coding_list = pickle.load(open('./count/coding_list.pkl', 'rb'))
        cluster_list = []
        for data_class in tqdm(range(cls.num_class)):
            coding = coding_list[data_class]
            cluster = KMeans(n_clusters=cls.num_cluster, random_state=random_state).fit(coding)
            cluster_list.append(cluster)
        pickle.dump(cluster_list, open('./count/kmeans_list.pkl', 'wb'))

    @classmethod
    def print_kmeans_info(cls):
        kmeans_list = pickle.load(open('./count/kmeans_list.pkl', 'rb'))
        for i in range(cls.num_class):
            kmeans = kmeans_list[i]
            count = Counter(kmeans.labels_)
            print(count)

    @classmethod
    def _get_cluster_index_list(cls, data_class):
        """
        获取某个类别聚类后的index列表
        :param data_class:
        :return:
        """
        cluster_list = pickle.load(open('./count/kmeans_list.pkl', 'rb'))
        cluster = cluster_list[data_class]

        index_list = [[] for _ in range(cls.num_cluster)]
        for i, v in enumerate(cluster.labels_):
            index_list[v].append(i)
        return index_list

    @classmethod
    def _get_ni_info(cls, data_class, basic_cluster_id):
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

        # 获取聚类的数据 (4, num_sample, dim)
        # coding_cluster_list = []
        # for i in range(cls.num_cluster):
        #     coding_cluster = coding[cluster_index_list[0]]
        #     coding_cluster_list.append(coding_cluster.numpy())
        # coding_cluster_list = np.array(coding_cluster_list)

        # 取模长最大的为基类
        # cluster_centers = np.mean(coding_cluster_list, axis=1)
        # basic_cluster_id = np.argmax(np.linalg.norm(cluster_centers, ord=2, axis=1))
        # # basic_cluster_id = 1

        coding_basic_cluster = coding[cluster_index_list[basic_cluster_id]]

        ni_list = []
        for compared_cluster_index in cluster_index_list:
            coding_compared_cluster = coding[compared_cluster_index]
            ni = _ni_index(coding_basic_cluster, coding_compared_cluster)
            ni_list.append(ni)

        ni_rank = sorted(range(len(ni_list)), key=lambda k: ni_list[k])
        return ni_list, ni_rank

    @classmethod
    def _get_basic_cluster_id(cls, data_class):
        ni_list = []
        for id in range(cls.num_cluster):
            ni, _ = cls._get_ni_info(data_class, basic_cluster_id=id)
            ni_list.append(ni)
        # print(np.array(ni_list))
        # print()
        gap_list = []
        for ni in ni_list:
            ni = sorted(ni)
            gap = 0
            for i in range(1, len(ni)):
                gap += (ni[i] - ni[i - 1])
            gap_list.append(gap)
        # print(gap_list)

        return np.argmax(gap_list)

    @classmethod
    def print_ni_info(cls):
        ni_list = []

        for data_class in range(cls.num_class):
            basic_cluster_id = cls._get_basic_cluster_id(data_class)
            ni, rank = cls._get_ni_info(data_class, basic_cluster_id)
            print(ni, rank)
            ni_list.append(sorted(ni))

        print(np.mean(np.array(ni_list), axis=0))

    @classmethod
    def celeba2txt(cls):
        # TODO
        root = './dataset/celeba_shift/4.txt'

        with open(root, 'w') as f:
            count = 0

            for data_class in range(cls.num_class):
                # 获取某个数字类别所有的在盖类别内的聚类index
                cluster_index_list = cls._get_cluster_index_list(data_class)
                # 得到聚类的NI值的排名
                basic_cluster_id = cls._get_basic_cluster_id(data_class)
                _, rank = cls._get_ni_info(data_class, basic_cluster_id=basic_cluster_id)
                cluster_index = cluster_index_list[rank[0]]  # TODO
                count += len(cluster_index)
                print(count)

                img_list = cls._get_img_list(data_class)

                # 该聚类对应的img list
                img_cluster = []
                for idx in cluster_index:
                    img_cluster.append(img_list[idx])

                img_cluster = [f'{img} {label}\n' for img, label in img_cluster]
                f.writelines(img_cluster)


class AttributeAbout:
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
    def plot_img(cls, attr_id_list=(), attr_mode_list=()):
        root = '../dataset/CelebA/Img/img_align_celeba'

        img_list = cls._get_attrs_list(attr_id_list, attr_mode_list)
        for img in img_list[:5]:
            plt.imshow(Image.open(f'{root}/{img}'))
            plt.show()

    @classmethod
    def print_hair_info_with_attr(cls, basic_attr_id=3):
        """
        输出某种属性下不同头发颜色的数量
        :param basic_attr_id:
        :return:
        """
        with open(cls.root, 'r') as f:
            f.readline()
            attr_list = f.readline().split()

        pos = {}
        for attr_id in [9, 10, 12, 18]:
            attr_name = attr_list[attr_id - 1]
            res = cls._get_attrs_list(attr_id_list=(basic_attr_id, attr_id), attr_mode_list=(1, 1))
            pos[attr_name] = len(res)
        neg = {}
        for attr_id in [9, 10, 12, 18]:
            attr_name = attr_list[attr_id - 1]
            res = cls._get_attrs_list(attr_id_list=[basic_attr_id, attr_id], attr_mode_list=[-1, 1])
            neg[attr_name] = len(res)
        print(f"{attr_list[basic_attr_id - 1]}")
        print(f'pos: {pos}')
        print(f'neg: {neg}')

    @classmethod
    def celeba_shift2txt(cls):
        root = './dataset/celeba_shift'

        male_id = 21

        blackhair_id = 9
        blondhair_id = 10
        brownhair_id = 12
        grayhair_id = 18

        num_examples = 1200

        # origin
        male = (
            cls._get_attrs_list((male_id, blackhair_id), (1, 1))[:num_examples],
            cls._get_attrs_list((male_id, blondhair_id), (1, 1))[:num_examples],
            cls._get_attrs_list((male_id, brownhair_id), (1, 1))[:num_examples],
            cls._get_attrs_list((male_id, grayhair_id), (1, 1))[:num_examples]
        )

        female = (
            cls._get_attrs_list((male_id, blackhair_id), (-1, 1))[:num_examples],
            cls._get_attrs_list((male_id, blondhair_id), (-1, 1))[:num_examples],
            cls._get_attrs_list((male_id, brownhair_id), (-1, 1))[:num_examples],
            cls._get_attrs_list((male_id, grayhair_id), (-1, 1))[:num_examples]
        )

        # origin
        with open(f'{root}/0.txt', 'w') as f:
            pos = [f'{img} 1\n' for img in np.hstack(male)]
            neg = [f'{img} 0\n' for img in np.hstack(female)]
            f.writelines(pos)
            f.writelines(neg)

        # marginal shift: P(X) change P(Y|X) remains
        with open(f'{root}/1.txt', 'w') as f:
            pos = male[0] + male[1]
            neg = female[0] + female[1]
            pos = [f'{img} 1\n' for img in pos]
            neg = [f'{img} 0\n' for img in neg]
            f.writelines(pos)
            f.writelines(neg)

        with open(f'{root}/4.txt', 'w') as f:
            pos = male[2] + male[3]
            neg = female[2] + female[3]
            pos = [f'{img} 1\n' for img in pos]
            neg = [f'{img} 0\n' for img in neg]
            f.writelines(pos)
            f.writelines(neg)

        # conditional shift: P(Y|X) change P(X) remains
        with open(f'{root}/2.txt', 'w') as f:
            pos = male[0] + male[2]
            neg = female[1] + female[3]

            pos = [f'{img} 1\n' for img in pos]
            neg = [f'{img} 0\n' for img in neg]

            f.writelines(pos)
            f.writelines(neg)

        with open(f'{root}/5.txt', 'w') as f:
            pos = male[1] + male[3]
            neg = female[0] + female[2]

            pos = [f'{img} 1\n' for img in pos]
            neg = [f'{img} 0\n' for img in neg]

            f.writelines(pos)
            f.writelines(neg)

        # joint shift: P(X) change P(Y|X) change
        with open(f'{root}/3.txt', 'w') as f:
            pos = male[0] + male[1]
            neg = female[1] + female[3]

            pos = [f'{img} 1\n' for img in pos]
            neg = [f'{img} 0\n' for img in neg]

            f.writelines(pos)
            f.writelines(neg)

        with open(f'{root}/6.txt', 'w') as f:
            pos = male[2] + male[3]
            neg = female[0] + female[2]

            pos = [f'{img} 1\n' for img in pos]
            neg = [f'{img} 0\n' for img in neg]

            f.writelines(pos)
            f.writelines(neg)

    @classmethod
    def celeba_conditional2txt(cls):
        def f():
            os.makedirs(data_dir, exist_ok=True)
            with open(f'{data_dir}/train.txt', 'w') as f:
                pos, neg = [], []
                for i in range(4):
                    pos_split = int(num_examples * p[i])
                    neg_split = int(num_examples * (1 - p[i]))
                    pos += male[i][:pos_split]
                    neg += female[i][:neg_split]
                pos = [f'{img} 1\n' for img in pos]
                neg = [f'{img} 0\n' for img in neg]
                f.writelines(pos)
                f.writelines(neg)

            with open(f'{data_dir}/test.txt', 'w') as f:
                pos, neg = [], []
                for i in range(4):
                    pos_split = int(num_examples * p[i])
                    neg_split = int(num_examples * (1 - p[i]))
                    pos += male[i][pos_split:]
                    neg += female[i][neg_split:]
                pos = [f'{img} 1\n' for img in pos]
                neg = [f'{img} 0\n' for img in neg]
                f.writelines(pos)
                f.writelines(neg)

        root = './dataset/celeba_correlation'

        male_id = 21

        blackhair_id = 9
        blondhair_id = 10
        brownhair_id = 12
        grayhair_id = 18

        num_examples = 1200

        # origin
        male = (
            cls._get_attrs_list((male_id, blackhair_id), (1, 1))[:num_examples],
            cls._get_attrs_list((male_id, blondhair_id), (1, 1))[:num_examples],
            cls._get_attrs_list((male_id, brownhair_id), (1, 1))[:num_examples],
            cls._get_attrs_list((male_id, grayhair_id), (1, 1))[:num_examples]
        )

        female = (
            cls._get_attrs_list((male_id, blackhair_id), (-1, 1))[:num_examples],
            cls._get_attrs_list((male_id, blondhair_id), (-1, 1))[:num_examples],
            cls._get_attrs_list((male_id, brownhair_id), (-1, 1))[:num_examples],
            cls._get_attrs_list((male_id, grayhair_id), (-1, 1))[:num_examples]
        )

        p = (0.5, 0.5, 0.5, 0.5)
        data_dir = f'{root}/0'
        f()

        p = (0.6, 0.4, 0.6, 0.4)
        data_dir = f'{root}/1'
        f()

        p = (0.7, 0.3, 0.7, 0.3)
        data_dir = f'{root}/2'
        f()

        p = (0.8, 0.2, 0.8, 0.2)
        data_dir = f'{root}/3'
        f()

        p = (0.9, 0.1, 0.9, 0.1)
        data_dir = f'{root}/4'
        f()

        p = (1, 0, 1, 0)
        data_dir = f'{root}/5'
        f()

    @classmethod
    def celeba_marginal2txt(cls):
        def f():
            os.makedirs(data_dir, exist_ok=True)
            with open(f'{data_dir}/train.txt', 'w') as f:
                pos, neg = [], []
                for i in range(4):
                    split = int(num_examples * p[i])
                    pos += male[i][:split]
                    neg += female[i][:split]
                pos = [f'{img} 1\n' for img in pos]
                neg = [f'{img} 0\n' for img in neg]
                f.writelines(pos)
                f.writelines(neg)

            with open(f'{data_dir}/test.txt', 'w') as f:
                pos, neg = [], []
                for i in range(4):
                    split = int(num_examples * p[i])
                    pos += male[i][split:]
                    neg += female[i][split:]
                pos = [f'{img} 1\n' for img in pos]
                neg = [f'{img} 0\n' for img in neg]
                f.writelines(pos)
                f.writelines(neg)

        root = './dataset/celeba_diversity'

        male_id = 21

        blackhair_id = 9
        blondhair_id = 10
        brownhair_id = 12
        grayhair_id = 18

        num_examples = 1200

        # origin
        male = (
            cls._get_attrs_list((male_id, blackhair_id), (1, 1))[:num_examples],
            cls._get_attrs_list((male_id, blondhair_id), (1, 1))[:num_examples],
            cls._get_attrs_list((male_id, brownhair_id), (1, 1))[:num_examples],
            cls._get_attrs_list((male_id, grayhair_id), (1, 1))[:num_examples]
        )

        female = (
            cls._get_attrs_list((male_id, blackhair_id), (-1, 1))[:num_examples],
            cls._get_attrs_list((male_id, blondhair_id), (-1, 1))[:num_examples],
            cls._get_attrs_list((male_id, brownhair_id), (-1, 1))[:num_examples],
            cls._get_attrs_list((male_id, grayhair_id), (-1, 1))[:num_examples]
        )

        p = (0.5, 0.5, 0.5, 0.5)
        data_dir = f'{root}/0'
        f()

        p = (0.9, 0.7, 0.3, 0.1)
        data_dir = f'{root}/1'
        f()

        p = (1, 0.9, 0.1, 0)
        data_dir = f'{root}/2'
        f()

        p = (1, 1, 0, 0)
        data_dir = f'{root}/3'
        f()

    @classmethod
    def celeba_cluster2txt(cls):
        random.seed(2)
        # 聚类
        with open('./count/img4cluster.txt', 'w') as f:
            attr_id = 3  # Attractive
            for label in [0, 1]:
                mode = -1 if label == 0 else 1

                img_list = cls._get_attrs_list((attr_id,), (mode,))
                img_list = random.sample(img_list, 25000)

                img_list = [f'{_} {label}\n' for _ in img_list]
                f.writelines(img_list)


def plot(img, root='../dataset/CelebA/Img/img_align_celeba'):
    img = Image.open(f'{root}/{img}')
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    AttributeAbout.celeba_marginal2txt()
