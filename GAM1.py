'''
使用验证集调整参数，选择模型，评价标准为precision和f1 score

'''


from Data import DATA
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.linear_model import Perceptron
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import numpy as np
import random
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from pygam import LogisticGAM
from sampling_for_imbalance import re_sampling
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,fbeta_score
from Feature_Processing import Feature_Scaling,Feature_Discretizer
import warnings
warnings.filterwarnings('ignore')
fc = Feature_Scaling()
fd = Feature_Discretizer()
data = DATA()


class GAM(object):

    def __init__(self):

        event = data.step1_data_train()

        self.data_all1, self.label1 = event[:, :-1], event[:, -1]

        self.conf_dict = None

    def update_settings(self, conf_dict):

        self.conf_dict = conf_dict

    def train(self):
        val_event = data.step1_data_test()
        val_data, val_label = val_event[:, :-1], val_event[:, -1]

        model = LogisticGAM().gridsearch(self.data_all1, self.label1)
        pre1_1 = model.predict(val_data)
        #print(self.metric(pre1_1, val_label))
        return pre1_1



    def metric(self,pre,test_label):
        pre = fc.pre_deal(pre)
        f1 = f1_score(test_label, pre)
        f05 = fbeta_score(test_label, pre, beta=0.5)
        reca = recall_score(test_label, pre)
        prec = precision_score(test_label, pre)
        accuracy = accuracy_score(test_label, pre)
        return f1,reca,prec,accuracy



if __name__ == '__main__':

    model_boosting = GAM()
    model_boosting.train()






