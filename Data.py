import numpy as np
import os
import random
from Feature_Processing import Feature_Scaling,Feature_Discretizer


class DATA(object):
    def __init__(self):

        self.PATH_1 = './data/All/partition7/NPY_DATA/'
        self.PATH_test_1 = "./data/All/partition7/NPY_test/"
        self.PATH_val_1 = './data/For_Train/NPY_val/'

        self.fd = Feature_Discretizer()
        self.fc = Feature_Scaling()


    def __iter__(self):
        while(1):
            x,y =  self.gen_batch_data()
            yield (x,y)


    def update_dataset(self,X,Y):
        self.X = X
        self.Y = Y
        self.idx_lis = list(range(self.X.shape[0]))
        self.epoch_it_num = self.X.shape[0]


    def gen_batch_data(self):
        idx = random.randint(0, self.epoch_it_num - 1)#随机选择一个索引，范围在0和self.epoch_it_num - 1之间。
        x = np.array([self.X[idx]])
        y = np.array([self.Y[idx]])
        y = np.reshape(y, (-1,1))
        return x, y


    def step1_processing(self,event_lis,data_type = "all"):
        data_train = self.flatten_event(self.get_all_event())
        if data_type == "new_feature":
            m = event_lis.shape[0]
            data = np.ones((m, 21))
            data[:, :10] = self.fc.normalization(event_lis[:, :-1], data_train[:, :-1])
            data[:, 10:20] = self.fc.new_feature(event_lis[:, :-1], data_train[:, :-1])
            data[:, -1] = event_lis[:, -1]
        else:
            #print("-----event_lis.shape",event_lis.shape)
            m = event_lis.shape[0]
            data = np.ones((m, 11))
            data[:, :10] = self.fc.normalization(event_lis[:, :-1], data_train[:, :-1])
            data[:, -1] = event_lis[:, -1]
        return data



    def step1_data_train(self,data_type = "all"):
        if data_type == "new_feature":
            event_lis = self.flatten_event(self.new_label_event())
            data = self.step1_processing(event_lis,data_type = "new_feature")
        else:
            event_lis = self.flatten_event(self.new_label_event())
            data = self.step1_processing(event_lis,data_type = "all")
        return data


    def step1_data_test(self,data_type = "all"):
        if data_type == "new_feature":
            event_lis = self.flatten_event(self.new_label_test())
            data = self.step1_processing(event_lis, data_type="new_feature")
        else:
            event_lis = self.flatten_event(self.new_label_test())
            data = self.step1_processing(event_lis,data_type = "all")
        return data

    def step1_data_val(self,data_type = "all"):
        if data_type == "new_feature":
            event_lis = self.flatten_event(self.new_label_val())
            data = self.step1_processing(event_lis, data_type="new_feature")
        else:
            event_lis = self.flatten_event(self.new_label_val())
            data = self.step1_processing(event_lis, data_type="all")
        return data


    def get_all_event_val(self):
        event_lis = []
        DIR_PATH = self.PATH_val_1
        file_lis = os.listdir(DIR_PATH)
        for f in file_lis:
            file_path = DIR_PATH + f
            npy_file = np.load(file_path)
            event_lis.append(npy_file)
        event_lis = np.array(event_lis)
        return event_lis




    def get_all_event_test(self):
        event_lis = []
        DIR_PATH = self.PATH_test_1
        file_lis = os.listdir(DIR_PATH)
        for f in file_lis:
            file_path = DIR_PATH + f
            npy_file = np.load(file_path)
            event_lis.append(npy_file.copy())
        event_lis = np.array(event_lis)
        return event_lis



    def get_all_event(self):
        '''
        按事件获取标准化后的数据
        '''
        event_lis = []
        file_lis = os.listdir(self.PATH_1)
        for f in file_lis:
            file_path = self.PATH_1 + f
            npy_file = np.load(file_path)
            n = npy_file.shape[0]
            if n > 1:
                event_lis.append(npy_file.copy())
        event_lis = np.array(event_lis)

        return event_lis

    def new_label_val(self):
        event_lis = []
        DIR_PATH = self.PATH_val_1
        file_lis = os.listdir(DIR_PATH)
        for f in file_lis:
            file_path = DIR_PATH + f
            npy_file = np.load(file_path)
            n = npy_file.shape[0]
            if sum(npy_file[:, 10]) > 0:
                npy_file[:, 10] = list(np.zeros(n) + 1)
            else:
                npy_file[:, 10] = list(np.zeros(n))
            event_lis.append(npy_file)
        event_lis = np.array(event_lis)
        return event_lis

    def new_label_test(self):
        event_lis = []
        DIR_PATH = self.PATH_test_1
        file_lis = os.listdir(DIR_PATH)
        for f in file_lis:
            file_path = DIR_PATH + f
            npy_file = np.load(file_path)
            n = npy_file.shape[0]
            if sum(npy_file[:, 10]) > 0:
                npy_file[:, 10] = list(np.zeros(n) + 1)
            else:
                npy_file[:, 10] = list(np.zeros(n))
            event_lis.append(npy_file)
        event_lis = np.array(event_lis)
        return event_lis

    def new_label_event(self):
        event_lis = []
        DIR_PATH = self.PATH_1
        file_lis = os.listdir(self.PATH_1)
        for f in file_lis:
            file_path = DIR_PATH + f
            npy_file = np.load(file_path)
            n = npy_file.shape[0]
            if n > 1:
                if sum(npy_file[:, 10]) > 0:
                    npy_file[:, 10] = list(np.zeros(n) + 1)
                else:
                    npy_file[:, 10] = list(np.zeros(n))
                event_lis.append(npy_file)
        event_lis = np.array(event_lis)
        return event_lis

    def flatten_event(self, event_lis):
        data = [None] * (len(event_lis) * 10000)
        idx = 0
        for k in range(event_lis.shape[0]):
            event = event_lis[k]
            for j in range(event.shape[0]):
                data[idx] = event[j]
                idx = idx + 1

        data = data[:idx]
        data = np.array(data)
        #print("data.shape",data.shape)
        
        return data


if __name__ == '__main__':
    data = DATA()



    