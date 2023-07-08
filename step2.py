from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC,OneClassSVM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
import numpy as np
from Data import DATA
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
from Metric import HSS,TSS

from Feature_Processing import Feature_Scaling,Feature_Discretizer
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sampling_for_imbalance import re_sampling  #采样
from pygam import LogisticGAM
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
fc = Feature_Scaling()
fd = Feature_Discretizer()
class Model_BOOSTING(object):

    def __init__(self):
        self.data = DATA()

        self.classifier_lis = [

            # LogisticRegressionCV(random_state=0), \
            LinearDiscriminantAnalysis(), \
            # QuadraticDiscriminantAnalysis(), \
            # SVC(kernel='linear',random_state=0),
            # GaussianNB(),
            # DecisionTreeClassifier(random_state=0), \
            #
            # ExtraTreesClassifier(random_state=0), \
            # BaggingClassifier(random_state=0), \
            RandomForestClassifier(random_state=0), \


            # BalancedBaggingClassifier(random_state=0),
            # BalancedRandomForestClassifier(random_state=0),
            AdaBoostClassifier(random_state=0), \
            # GradientBoostingClassifier(random_state=0), \
            RUSBoostClassifier(random_state=0,n_estimators=200),
            EasyEnsembleClassifier(random_state=0),
            ]

    def get_data_1(self, all, mean):
        which_1 = np.where(mean[:, -1] == 1)
        #获取label为1的索引
        y_1 = self.data.flatten_event(all)[which_1][:, -1]
        x_1 = self.data.flatten_event(all)[which_1][:, :-1]
        return x_1, y_1

    def train(self):
        all_event = self.data.get_all_event()
        all_data = self.data.flatten_event(all_event)[:, :-1]
        new_label = self.data.flatten_event(self.data.new_label_event())
        x_1, y_1 = self.get_data_1(all_event, new_label)

        val_event = self.data.get_all_event_test()
        new_label_val = self.data.flatten_event(self.data.new_label_test())
        val_data, val_label = self.get_data_1(val_event, new_label_val)

        val_data = fc.standardization(val_data, all_data)

        test_event = self.data.get_all_event_test()
        new_label_test = self.data.flatten_event(self.data.new_label_test())
        test_data, test_label = self.data.flatten_event(test_event)[:, :-1], self.data.flatten_event(test_event)[:, -1]
        test_data = fc.standardization(test_data, all_data)

        x_1 = fc.standardization(x_1, all_data)

        constraints = [None, None, "concave", None, "concave", "concave", "concave", "concave", "concave", None]
        model_1 = LogisticGAM(constraints=constraints).gridsearch(x_1, y_1)
        model_2 = RUSBoostClassifier(random_state=0)
        # model_3 = EasyEnsembleClassifier(random_state=0)
        # model_4 = AdaBoostClassifier(random_state=0)
        # model_5 = RandomForestClassifier()
        model_6 = LinearDiscriminantAnalysis()

        model_1.fit(x_1, y_1)
        model_2.fit(x_1, y_1)
        # model_3.fit(x_1, y_1)
        # model_4.fit(x_1, y_1)
        # model_5.fit(x_1, y_1)
        model_6.fit(x_1, y_1)

        pre1_1 = model_1.predict(val_data)
        pre1_2 = model_2.predict(val_data)
        # pre1_3 = model_3.predict(val_data)
        # pre1_4 = model_4.predict(val_data)
        # pre1_5 = model_5.predict(val_data)
        pre1_6 = model_6.predict(val_data)

        print(self.metric(pre1_1, val_label))
        # print(self.metric(pre1_2, val_label))

        n = len(pre1_1)
        pre1 = []
        for i in range(n):
            if pre1_1[i]  + pre1_2[i]  + pre1_6[i]> 0:
                pre1.append(1)
            else:
                pre1.append(0)

        f1, reca, prec, accuracy = self.metric(pre1, val_label)
        print(f1, reca, prec, accuracy)

        return  pre1

    def metric(self, pre, test_label):
        pre = fc.pre_deal(pre)
        f1 = f1_score(test_label, pre)
        reca = recall_score(test_label, pre)
        prec = precision_score(test_label, pre)
        accuracy = accuracy_score(test_label, pre)
        return f1, reca, prec, accuracy

if __name__ == '__main__':
    model_boosting = Model_BOOSTING()
    model_boosting.train()