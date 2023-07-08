# ----------------------
# 先将数据按照事件进行分类
# 按事件划分训练集测试集
# ----------------------


import numpy as np
from astropy.io import fits
import os
import warnings
import copy
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# 用于将数据分离成验证集和测试集
class Regroup(object):
    def __init__(self):
        pass

    def normalize(self,data, max, min):
        dst = (data - min)  / (max - min)
        return dst

    def standard(self,data, mean, std):
        dst = (data - mean)  / std
        return dst

    def regroup(self, TXT_DATA_PATH, TXT_LABEL_PATH, NPY_PATH):
        '''
        读取fit文件并以npy形式保存至文件夹中,并以事件命名
        '''

        data_all_lis = np.zeros((80000, 11))
        data_lis = [None] * 10000

        f_data = open(TXT_DATA_PATH)
        f_label = open(TXT_LABEL_PATH)

        idx = 0
        idxnan = 0
        while True:
            data_lines = f_data.readline()  # 整行读取数据
            label_lines = f_label.readline()  # 整行读取数据
            if not label_lines:
                break

            data_lines = data_lines.split()
            label_lines = label_lines.split()
            data_lines.append(label_lines[-1])  # 如果DATA与LABEL的样本顺序不一致，这样操作是错误的，应该先按id排序，最准确的是按照ID进行merge操作

            file_name = data_lines[0].split('.')

            event_id = int(file_name[2])
            event_data = data_lines[1:]
            for k in range(len(event_data)):
                event_data[k] = float(event_data[k])


            if np.isnan(np.mean(event_data)):
                idxnan = idxnan + 1
                continue

            if data_lis[event_id] is None:
                data_lis[event_id] = []


            data_lis[event_id].append(event_data)

            data_all_lis[idx] = np.array(event_data)
            idx = idx + 1

        data_all_lis = data_all_lis[:idx]
        data_all_lis = np.array(data_all_lis)
        data_all_mean = np.mean(data_all_lis, axis=0)
        data_all_std = np.sqrt(np.var(data_all_lis, axis=0))

        data_all_max = np.max(data_all_lis, axis=0)
        data_all_min = np.min(data_all_lis, axis=0)


        data_all_mean[-1] = 0
        data_all_std[-1] = 1

        # np.save('mean', data_all_mean)
        # np.save('std', data_all_std)
        #
        # np.save('max', data_all_max)
        # np.save('min', data_all_min)

#原始数据
        for k in range(len(data_lis)):
            if data_lis[k] is not None:
                event_data = data_lis[k]
                event_data = np.array(event_data)
                np.save(NPY_PATH + str(k), event_data)

        print(idxnan)
#标准化
        # for k in range(len(data_lis)):
        #     if data_lis[k] is not None:
        #         event_data = data_lis[k]
        #         event_data = np.array(event_data)
        #         event_data = self.standard(event_data,data_all_mean,data_all_std)
        #         np.save(NPY_PATH + str(k), event_data)
        # print(idxnan)
# 归一化
#         for k in range(len(data_lis)):
#             if data_lis[k] is not None:
#                 event_data = data_lis[k]
#                 event_data = np.array(event_data)
#                 event_data = self.normalize(event_data,data_all_max,data_all_min)
#                 np.save(NPY_PATH + str(k), event_data)
#
#         print(idxnan)


if __name__ == '__main__':
    r_v = Regroup()
    #train
    # TXT_DATA_PATH = 'data/raw/train_para_input.txt'
    # TXT_LABEL_PATH = 'data/raw/train_output.txt'
    # NPY_PATH = 'data/standardization/'
    # r_v.regroup(TXT_DATA_PATH, TXT_LABEL_PATH, NPY_PATH)
    #test
    TXT_DATA_PATH = 'data/raw/test_para_input.txt'
    TXT_LABEL_PATH = 'data/raw/test_output.txt'
    NPY_PATH = 'data/NPY_test/'
    r_v.regroup(TXT_DATA_PATH, TXT_LABEL_PATH, NPY_PATH)
    #
    # TXT_DATA_PATH = 'data/raw/val_para_input.txt'
    # TXT_LABEL_PATH = 'data/raw/val_output.txt'
    # NPY_PATH = 'data/NPY_val_standardization/'
    # r_v.regroup(TXT_DATA_PATH, TXT_LABEL_PATH, NPY_PATH)


    #train
    # TXT_DATA_PATH = 'data/raw/train_split/train_input.txt'
    # TXT_LABEL_PATH = 'data/raw/train_split/train_output.txt'
    # NPY_PATH = 'data/For_Train/NPY_train/'
    # r_v.regroup(TXT_DATA_PATH, TXT_LABEL_PATH, NPY_PATH)
    #val
    TXT_DATA_PATH = 'data/raw/train_split/val_input.txt'
    TXT_LABEL_PATH = 'data/raw/train_split/val_output.txt'
    NPY_PATH = 'data/For_Train/NPY_val/'
    r_v.regroup(TXT_DATA_PATH, TXT_LABEL_PATH, NPY_PATH)

