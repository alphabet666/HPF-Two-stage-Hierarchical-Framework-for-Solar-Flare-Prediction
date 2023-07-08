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
from pygam import LogisticGAM
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from sampling_for_imbalance import re_sampling
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,fbeta_score
from Feature_Processing import Feature_Scaling,Feature_Discretizer
import warnings
warnings.filterwarnings('ignore')
fc = Feature_Scaling()
fd = Feature_Discretizer()
data = DATA()
samp1 = re_sampling(method = "NeighbourhoodCleaningRule")
samp2 = re_sampling(method = "NeighbourhoodCleaningRule")

class Model_BOOSTING(object):

    def __init__(self):

        event = data.step1_data_train()


        self.data_all1, self.label1 = event[:, :-1], event[:, -1]
        self.data_all2, self.label2 = samp1.fit_resample(event[:, :-1], event[:, -1])

        new_event = data.step1_data_train(data_type="new_feature")
        self.data_all3, self.label3 = new_event[:, :-1], new_event[:, -1]

        self.conf_dict = None

    def update_settings(self, conf_dict):
            self.conf_dict = conf_dict

    def train(self):
        model_1 = imb_xgb(special_objective='weighted', imbalance_alpha=0.945)
        model_2 = BalancedRandomForestClassifier(random_state=0, n_estimators=60,max_samples=2 )
        model_3 = QuadraticDiscriminantAnalysis(priors=[0.95,0.05])
        model_4 = GaussianNB()
        # model_5 = LogisticGAM().gridsearch(self.data_all3, self.label3)

        model_1.fit(self.data_all1, self.label1)
        model_2.fit(self.data_all2, self.label2)
        model_3.fit(self.data_all1, self.label1)
        model_4.fit(self.data_all2, self.label2)
        # model_5.fit(self.data_all2, self.label2)

        val_event = data.step1_data_test()
        val_data, val_label = val_event[:, :-1], val_event[:, -1]

        val_new = data.step1_data_test(data_type="new_feature")
        val_x1, val_y1 = val_new[:, :-1], val_new[:, -1]

        pre1_1 = model_1.predict_determine(val_data)
        pre1_2 = model_2.predict(val_data)
        pre1_3 = model_3.predict(val_data)
        pre1_4 = model_4.predict(val_data)
        # pre1_5 = model_5.predict(val_x1)

        p1_1 = model_1.predict_sigmoid(val_data)
        p1_2 = model_2.predict_proba(val_data)[:,1]
        p1_3 = model_3.predict_proba(val_data)[:,1]
        p1_4 = model_4.predict_proba(val_data)[:,1]
        # p1_5 = model_5.predict_proba(val_x1)

        # pre = p1_5

        # print("*"*20)
        # print(self.metric(pre1_1, val_label))
        # print(self.metric(pre1_2, val_label))
        # print(self.metric(pre1_3, val_label))
        # print(self.metric(pre1_4, val_label))
        # print(self.metric(pre1_5, val_label))

        n = len(pre1_1)
        pre1 = []
        for i in range(n):
            if p1_2[i] > 0.55   and  p1_4[i] > 0.55:
                pre1.append(1)
            else:
                pre1.append(0)

        f1,f05,reca,prec,accuracy = self.metric(pre1, val_label)
        # print("***"*10)
        # print(f1,f05,reca,prec,accuracy )

        return pre1


    def metric(self,pre,test_label):
        pre = fc.pre_deal(pre)
        f1 = f1_score(test_label, pre)
        f05 = fbeta_score(test_label, pre,beta=0.5)
        reca = recall_score(test_label, pre)
        prec = precision_score(test_label, pre)
        accuracy = accuracy_score(test_label, pre)
        return f1, f05,reca,prec,accuracy



if __name__ == '__main__':

    model_boosting = Model_BOOSTING()
    model_boosting.train()






