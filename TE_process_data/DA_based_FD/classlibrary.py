import numpy as np
import pandas as pd
import os

from sklearn.manifold import TSNE
from torch.autograd import Variable
from torch import nn
from torch.utils.data import Dataset
import torch
from sklearn.preprocessing import StandardScaler
import itertools
import matplotlib.pylab as plt

NO_LABEL = -1
class classLibrary():
    def __init__(self):
        '''一些必要的函数'''

    def exp_rampup(self, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        def warpper(epoch):
            if epoch < rampup_length:
                epoch = np.clip(epoch, 0.0, rampup_length)
                phase = 1.0 - epoch / rampup_length
                return float(np.exp(-5.0 * phase * phase))
            else:
                return 1.0
        return warpper

    def cosine_rampdown(self, rampdown_length, num_epochs):
        """Cosine rampdown from https://arxiv.org/abs/1608.03983"""

        def warpper(epoch):
            if epoch >= (num_epochs - rampdown_length):
                ep = .5 * (epoch - (num_epochs - rampdown_length))
                return float(.5 * (np.cos(np.pi * ep / rampdown_length) + 1))
            else:
                return 1.0

        return warpper

    def exp_rampdown(self, rampdown_length, num_epochs):
        """Exponential rampdown from https://arxiv.org/abs/1610.02242"""

        def warpper(epoch):
            if epoch >= (num_epochs - rampdown_length):
                ep = .5 * (epoch - (num_epochs - rampdown_length))
                return float(np.exp(-(ep * ep) / rampdown_length))
            else:
                return 1.0

        return warpper

    def encode_label(self, label):
        return NO_LABEL*(label +1)

    def decode_label(self, label):
        return NO_LABEL * label -1

    def exp_warmup(self, rampup_length, rampdown_length, num_epochs):
        rampup = self.exp_rampup(rampup_length)
        rampdown = self.exp_rampdown(rampdown_length, num_epochs)

        def warpper(epoch):
            return rampup(epoch) * rampdown(epoch)

        return warpper

    # TE数据的训练集和测试集的整理
    def data_conca(self, file_dirs, time_len, unlist, num_class):
        '''
        file_dirs: 数据路径
        time_len： 时间步（窗）
        unlist： 不进行分类的类别
        '''
        num_dataset = 600
        unlist = set(unlist)
        # 标准化操作，需要拿到基准数据的均值和方差
        standar = StandardScaler()
        if file_dirs=='D:/Python/DA_based_FD/matlab_pro/mode1_new/':
            fault0 = pd.read_csv(
                r'D:\Python\DA_based_FD\matlab_pro\mode1_new\\mode1_d00.csv')
        elif file_dirs=='D:/Python/DA_based_FD/matlab_pro/mode3_new/':
            fault0 = pd.read_csv(
                r'D:\Python\DA_based_FD\matlab_pro\mode3_new\\mode3_d00.csv')
        elif file_dirs=='D:/Python/GVB-master/GVB-GD/dataload/matlab_pro/mode2_new/':
            fault0 = pd.read_csv(
                r'D:\Python\GVB-master\GVB-GD\dataload\matlab_pro\mode2_new\\mode2_d00.csv')
        elif file_dirs=='D:/Python/GVB-master/GVB-GD/dataload/matlab_pro/mode4_new/':
            fault0 = pd.read_csv(
                r'D:\Python\GVB-master\GVB-GD\dataload\matlab_pro\mode4_new\\mode4_d00.csv')
        elif file_dirs=='D:/Python/GVB-master/GVB-GD/dataload/matlab_pro/mode6_new/':
            fault0 = pd.read_csv(
                r'D:\Python\GVB-master\GVB-GD\dataload\matlab_pro\mode6_new\\mode6_d00.csv')
        else :
            fault0 = pd.read_csv(
                r'D:\Python\GVB-master\GVB-GD\dataload\matlab_pro\mode5_new\\mode5_d00.csv')
        # fault0 = pd.read_csv(r'F:\Python程序\transferlearning-master\code\DeepDG\dataload\matlab_pro\mode4\\mode4_d00.csv')
        fault0 = np.array(fault0)
        # num_dataset_f0 = fault0.shape[0]
        standar.fit(fault0[1:num_dataset,:])
        # 数据集整合
        data_set = np.zeros(shape=(1, 10, 51))
        num = 0
        # 取文件夹所有文件
        for root, dirs, files in os.walk(file_dirs):
            for file in files:

                # 对单个文件进行处理
                file_dir = file_dirs + file
                data = pd.read_csv(file_dir)
                data = np.array(data)
                # 标准化
                # num_dataset = data.shape[0]
                data = standar.transform(data)
                data = np.concatenate(
                    (data[0:num_dataset, 0:45], data[0:num_dataset, 46:49], data[0:num_dataset, 50:52]), axis=1)
                label = [[num]] * len(data)
                label = np.array(label)
                data = np.concatenate((data, label), axis=1)
                data_temp = np.zeros(((len(data) - time_len + 1), time_len, data.shape[1]))
                if time_len > 1:
                    for i in range((len(data)-time_len+1)):
                        for j in range(time_len):
                            data_temp[i][j] = data[i+j]

                    # data_set.append(data_temp)
                    # data = np.concatenate((data_temp, label), axis=1)
                else:
                    # data = np.concatenate((data, label), axis=1)
                    data_set.append(data)
                data_set = np.concatenate((data_set, data_temp), axis=0)
                # data_set[(num_dataset-9) * num:(num_dataset-9) * (num + 1), :, :] = data_temp

                num = num+1
        data_set = data_set[1:data_set.shape[0], :, :]
        return data_set

    def data_conca_21(self, file_dirs, time_len, unlist, num_class):
        '''
        file_dirs: 数据路径
        time_len： 时间步（窗）
        unlist： 不进行分类的类别
        '''
        num_dataset = 600
        unlist = set(unlist)
        # 标准化操作，需要拿到基准数据的均值和方差
        standar = StandardScaler()
        if file_dirs==r'D:\Python\GVB-master\GVB-GD\dataload\TE_new\mode1_new\\':
            fault0 = pd.read_csv(
                r'D:\Python\GVB-master\GVB-GD\dataload\TE_new\mode1_new\mode1_d00.csv')
        elif file_dirs==r'D:\Python\GVB-master\GVB-GD\dataload\TE_new\mode3_new\\':
            fault0 = pd.read_csv(
                r'D:\Python\GVB-master\GVB-GD\dataload\TE_new\mode3_new\mode3_d00.csv')
        elif file_dirs==r'D:\Python\GVB-master\GVB-GD\dataload\TE_new\mode2_new\\':
            fault0 = pd.read_csv(
                r'D:\Python\GVB-master\GVB-GD\dataload\TE_new\mode2_new\mode2_d00.csv')
        elif file_dirs==r'D:\Python\GVB-master\GVB-GD\dataload\TE_new\mode4_new\\':
            fault0 = pd.read_csv(
                r'D:\Python\GVB-master\GVB-GD\dataload\TE_new\mode4_new\mode4_d00.csv')
        elif file_dirs==r'D:\Python\GVB-master\GVB-GD\dataload\TE_new\mode6_new\\':
            fault0 = pd.read_csv(
                r'D:\Python\GVB-master\GVB-GD\dataload\TE_new\mode6_new\mode6_d00.csv')
        else :
            fault0 = pd.read_csv(
                r'D:\Python\GVB-master\GVB-GD\dataload\TE_new\mode5_new\mode5_d00.csv')
        # fault0 = pd.read_csv(r'F:\Python程序\transferlearning-master\code\DeepDG\dataload\matlab_pro\mode4\\mode4_d00.csv')
        fault0 = np.array(fault0)
        # num_dataset_f0 = fault0.shape[0]
        standar.fit(fault0[1:num_dataset,:])
        # 数据集整合
        data_set = np.zeros(shape=(1, 10, 51))
        num = 0
        # 取文件夹所有文件
        for root, dirs, files in os.walk(file_dirs):
            for file in files:

                # 对单个文件进行处理
                file_dir = file_dirs + file
                data = pd.read_csv(file_dir)
                data = np.array(data)
                # 标准化
                # num_dataset = data.shape[0]
                data = standar.transform(data)
                data = np.concatenate(
                    (data[0:num_dataset, 0:45], data[0:num_dataset, 46:49], data[0:num_dataset, 50:52]), axis=1)
                label = [[num]] * len(data)
                label = np.array(label)
                data = np.concatenate((data, label), axis=1)
                data_temp = np.zeros(((len(data) - time_len + 1), time_len, data.shape[1]))
                if time_len > 1:
                    for i in range((len(data)-time_len+1)):
                        for j in range(time_len):
                            data_temp[i][j] = data[i+j]

                    # data_set.append(data_temp)
                    # data = np.concatenate((data_temp, label), axis=1)
                else:
                    # data = np.concatenate((data, label), axis=1)
                    data_set.append(data)
                data_set = np.concatenate((data_set, data_temp), axis=0)
                # data_set[(num_dataset-9) * num:(num_dataset-9) * (num + 1), :, :] = data_temp

                num = num+1
        data_set = data_set[1:data_set.shape[0], :, :]
        return data_set

    def data_conca_LS(self, file_dirs, time_len, unlist, num_class):
        '''
        file_dirs: 数据路径
        time_len： 时间步（窗）
        unlist： 不进行分类的类别
        '''
        # num_dataset = 600
        unlist = set(unlist)
        # 标准化操作，需要拿到基准数据的均值和方差
        standar = StandardScaler()
        if file_dirs==r'D:\GVB-master\GVB-GD\dataload\matlab_pro\mode1_LS\\':
            fault0 = pd.read_csv(
                r'D:\GVB-master\GVB-GD\dataload\matlab_pro\mode1_LS\\mode1_d00.csv')
        elif file_dirs==r'D:\GVB-master\GVB-GD\dataload\matlab_pro\mode3_LS\\':
            fault0 = pd.read_csv(
                r'D:\GVB-master\GVB-GD\dataload\matlab_pro\mode3_LS\\mode3_d00.csv')
        elif file_dirs==r'D:\GVB-master\GVB-GD\dataload\matlab_pro\mode2_LS\\':
            fault0 = pd.read_csv(
                r'D:\GVB-master\GVB-GD\dataload\matlab_pro\mode2_LS\\mode2_d00.csv')
        elif file_dirs==r'D:\GVB-master\GVB-GD\dataload\matlab_pro\mode4_LS\\':
            fault0 = pd.read_csv(
                r'D:\GVB-master\GVB-GD\dataload\matlab_pro\mode4_LS\\mode4_d00.csv')
        elif file_dirs==r'D:\GVB-master\GVB-GD\dataload\matlab_pro\mode6_LS\\':
            fault0 = pd.read_csv(
                r'D:\GVB-master\GVB-GD\dataload\matlab_pro\mode6_LS\\mode6_d00.csv')
        else :
            fault0 = pd.read_csv(
                r'D:\GVB-master\GVB-GD\dataload\matlab_pro\mode5_LS\\mode5_d00.csv')
        # fault0 = pd.read_csv(r'F:\Python程序\transferlearning-master\code\DeepDG\dataload\matlab_pro\mode4\\mode4_d00.csv')
        fault0 = np.array(fault0)
        num_dataset_f0 = fault0.shape[0]
        standar.fit(fault0[1:num_dataset_f0,:])
        # 数据集整合
        data_set = np.zeros(shape=(1, 10, 51))
        num = 0
        # 取文件夹所有文件
        for root, dirs, files in os.walk(file_dirs):
            for file in files:

                # 对单个文件进行处理
                file_dir = file_dirs + file
                data = pd.read_csv(file_dir)
                data = np.array(data)
                # 标准化
                num_dataset = data.shape[0]
                data = standar.transform(data)
                data = np.concatenate(
                    (data[0:num_dataset, 0:45], data[0:num_dataset, 46:49], data[0:num_dataset, 50:52]), axis=1)
                label = [[num]] * len(data)
                label = np.array(label)
                data = np.concatenate((data, label), axis=1)
                data_temp = np.zeros(((len(data) - time_len + 1), time_len, data.shape[1]))
                if time_len > 1:
                    for i in range((len(data)-time_len+1)):
                        for j in range(time_len):
                            data_temp[i][j] = data[i+j]

                    # data_set.append(data_temp)
                    # data = np.concatenate((data_temp, label), axis=1)
                else:
                    # data = np.concatenate((data, label), axis=1)
                    data_set.append(data)
                data_set = np.concatenate((data_set, data_temp), axis=0)
                # data_set[(num_dataset-9) * num:(num_dataset-9) * (num + 1), :, :] = data_temp

                num = num+1
        data_set = data_set[1:data_set.shape[0], :, :]
        return data_set

    def decode_targets(self, targets):
        label_mask = targets.ge(0)  # .ge(0)比较和0的大小,大于等于取1
        unlab_mask = targets.le(NO_LABEL)  # NO_LABEL = -1，小于等于取1
        targets[unlab_mask] = self.decode_label(targets[unlab_mask])  # decode_label = NO_LABEL * label -1
        return label_mask, unlab_mask

    def gen_info(self, info, lbs, ubs, iteration=True):
        ret = []
        nums = {'l': lbs, 'u': ubs, 'a': lbs + ubs, 'r':lbs + ubs, 't':lbs + ubs }
        for k, val in info.items():
            n = nums[k[0]]
            v = val[-1] if iteration else sum(val)
            s = f'{k}: {v / n:.3%}' if k[-1] == 'c' else f'{k}: {v:.5f}'
            ret.append(s)
        return '\t'.join(ret)

    def confusion_matrix(self, preds, labels, conf_matrix):
        for p, t in zip(preds, labels):
            conf_matrix[t, p] += 1
        return conf_matrix

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        '''
        this function prints and plots the confusion matrix.
        Normalization can be applied by setting 'normalize=True'.
        input
        - cm: 计算出的混淆矩阵的值
        - classes : 混淆矩阵中每一行每一列对应的列
        - normalize: true:显示百分比， false: 显示个数
        '''
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        # print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)       # classes，可以换成指定的列表,如 list = ['1', '2', '3']
        plt.yticks(tick_marks, classes)

        # 解决显示不全问题
        plt.axis("equal")
        ax = plt.gca()  # 获得当前axis
        left, right = plt.xlim()  # 获得x轴最大最小值
        ax.spines['left'].set_position(('data', left))
        ax.spines['right'].set_position(('data', right))
        for edge_i in ['top', 'bottom', 'right', 'left']:
            ax.spines[edge_i].set_edgecolor('white')

        #
        thresh = cm.max() / 2
        print("thresh", thresh.dtype)

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
            # print("num", num)
            plt.text(j, i, num,
                     verticalalignment='center',
                     horizontalalignment="center",
                     color="white" if float(num) > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    def visualize_scatter(self, data_2d, labels, every_label_sam, color):
        '''
        该函数用于批量绘画数据，并根据不同的批量来指定对应的标签
        :param data_2d: 二维坐标点（比如t-sne降维后的数据）
        :param labels: 需要绘制的不同颜色数据的标签
        :param every_label_sam: 每个标签对应的样本数量
        :return:无返回值
        '''
        label_to_id_dict = {v: i for i, v in enumerate(labels)}     # np.unique(labels)  去掉重复的数字并进行排序
        id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
        label_ids = np.array([label_to_id_dict[x] for x in labels])
        print(label_ids)
        figsize = (8, 8)
        # plt.figure(figsize=figsize)
        # plt.figure()
        nb_classes = len(label_ids)
        for label_id in label_ids:
            if label_id <= 7:
                plt.scatter(data_2d[label_id * every_label_sam:(label_id + 1) * every_label_sam, 0],
                            data_2d[label_id * every_label_sam:(label_id + 1) * every_label_sam, 1],
                            marker='o', s=10,
                            color=color[label_id],   # / float(nb_classes)
                            label=id_to_label_dict[label_id], )
            elif label_id <= 7*2:
                plt.scatter(data_2d[label_id * every_label_sam:(label_id + 1) * every_label_sam, 0],
                            data_2d[label_id * every_label_sam:(label_id + 1) * every_label_sam, 1],
                            marker='o', s=10,
                            color=color[label_id],  # / float(nb_classes)
                            label=id_to_label_dict[label_id], )
            else:
                plt.scatter(data_2d[label_id * every_label_sam:(label_id + 1) * every_label_sam, 0],
                            data_2d[label_id * every_label_sam:(label_id + 1) * every_label_sam, 1],
                            marker='o', s=10,
                            color=color[label_id],  # / float(nb_classes)
                            label=id_to_label_dict[label_id], )
        plt.xlabel('', fontdict={'size': 14})
        plt.xticks([])
        plt.yticks([])
        num1 = 1.13
        num2 = 0.8
        num4 = 0
        # plt.legend(loc='best', fontsize=10, bbox_to_anchor=(num1, num2), borderaxespad=num4)    # 控制图例的位置  num1：水平位置  num2：垂直位置

    def t_sne_func(self, data, labels, every_label_sam, color):
        '''labels = ['B007', 'B021', 'IR007', 'IR021', 'OR007', 'OR021', 'Normal']
        函数用来将数据用t-sne降维，再将降维之后的数据绘图
        :param data: 待降维的数据
        :param labels: 数据标签
        :param every_label_sam: 每个标签对应的样本数量
        :return:
        '''
        tsne = TSNE().fit_transform(data)
        scaler = StandardScaler()
        tsne = scaler.fit_transform(tsne)
        self.visualize_scatter(tsne, labels, every_label_sam, color)
        # plt.show()





class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """
    def __init__(self, xy, dim, transforms=None):
        xy = np.array(xy)
        if dim == 4:
            xy_x = xy[:, :, 0:-1]; xy_x = np.reshape(xy_x, (xy_x.shape[0], 1, xy_x.shape[1], xy_x.shape[2]))
        elif dim == 3:
            xy_x = xy[:, 0:-1]; xy_x = np.reshape(xy_x, (xy_x.shape[0], 1, xy_x.shape[1]))
        else:
            xy_x = xy[:, 0:-1]
        self.x_data = torch.from_numpy(xy_x)
        if dim == 4:
            self.y_data = torch.from_numpy(xy[:, 0, -1])
        else:
            self.y_data = torch.from_numpy(xy[:, -1])
        self.len = xy.shape[0]
        self.transform = transforms

    def __getitem__(self, index):
        if self.transform:
            self.x_data = self.transform(self.x_data)
            self.y_data = self.transform(self.y_data)
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class GaussianNoise(nn.Module):  # 添加高斯白噪声
    def __init__(self, std):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        zeros = torch.zeros(x.size()).cuda()
        n = Variable(torch.normal(zeros, std=self.std).cuda())
        return x + n