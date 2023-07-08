
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer,QuantileTransformer,PowerTransformer,Binarizer,LabelEncoder,KBinsDiscretizer
from feature_engine.discretisation import DecisionTreeDiscretiser
from scorecardbundle.feature_discretization.ChiMerge import ChiMerge
from sklearn.preprocessing import Binarizer,LabelEncoder,KBinsDiscretizer

import numpy as np
import pandas as pd

class Feature_Scaling(object):


    def standardization(self,data,data_train):
        mean_data = data_train.mean(0)
        std_data = data_train.std(0)
        data = (data - mean_data)/std_data
        return data

    def new_feature(self, data, data_train):
        mean_data = np.percentile(data_train,50)
        data = np.sign(data - mean_data)
        return data

    def normalization(self,data,data_train):
        min_data = data_train.min(0)
        max_data = data_train.max(0)
        data = (data - min_data)/(max_data - min_data)
        return data

    def normalization1(self,data,data_train):
        min_data = data_train.min(0)
        max_data = data_train.max(0)
        k = 2 / (max_data - min_data)
        data = 1 + k*(data - max_data )
        return data

    def pre_deal(self,pre):
        for i in range(len(pre)-2):
            if pre[i] < 1 and pre[i+2] < 1:
                pre[i + 1] = 0
            elif pre[i] > 0 and pre[i+2] > 0 :
                pre[i+1] = 1

            else:
                pre[i + 1] = pre[i + 1]
        return pre



class Feature_Discretizer(object):

    def feature_discretizer(self,data_m=None,data_d=None,method="binarizer",q=99):
        if method == "binarizer":
            return self.binarizer(data_m,data_d,q=q)
        if method == "kmeans_Discretizer":
            return self.kmeans_Discretizer(data_m,data_d)
        if method == "ChiMerge_Discretizer":
            return self.ChiMerge_Discretizer(data_m,data_d)
        if method == "DecisionTree_Discretizer":
            return self.DecisionTree_Discretizer(data_m,data_d)


    def kmeans_Discretizer(self, data_m=None,data=None):
        data = data[:, :-1]
        n = data.shape[1]
        for i in range(n):
            data[:, i] = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='kmeans').fit_transform(
                data[:, i].reshape(-1, 1)).reshape(1, -1)
        return data


    def ChiMerge_Discretizer(self, data_m=None,data=None):
        n = data.shape[1] - 1
        for i in range(n):
            data[:, i] = LabelEncoder().fit_transform(
                ChiMerge().fit_transform(X=data[:, i].reshape(-1, 1), y=data[:, -1])).reshape(1, -1)
        return data[:, :-1]


    def DecisionTree_Discretizer(self, data_m,data_d):
        n = data_m.shape[1] - 1
        for i in range(n):
            mo = DecisionTreeDiscretiser(cv=5,regression = True,random_state =0).fit(X=pd.DataFrame(data_m[:, i]), y=pd.Series(data_m[:, -1]))
            data_d[:, i] = mo.transform(X=pd.DataFrame(data_d[:, i])).values.reshape(1, -1)
        return data_d[:, :-1]

    def binarizer(self,data_m=None,data=None,q=99):
        data = data[:, :-1]
        threshold = np.percentile(data, q=q, axis=0)
        data = Binarizer(threshold = threshold).fit_transform(data)
        return data

if __name__ == '__main__':
    from Data import DATA
    data = DATA()
    event_lis = data.flatten_event(data.new_label_event())
    fc = Feature_Scaling()
    print(fc.new_feature(event_lis[:,:-1],event_lis[:,:-1]))





