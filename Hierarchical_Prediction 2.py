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
from sklearn.neural_network import MLPClassifier
from Feature_Processing import Feature_Scaling
data = DATA()
fc = Feature_Scaling()

#from base_model_step2 import BASE_MODEL_BOOSTING as BOOSTING_2
# from step1 import Model_BOOSTING as BOOSTING_1
from step1 import Model_BOOSTING as BOOSTING_1

from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier
import pandas as pd
from sampling_for_imbalance import re_sampling  #采样
samp1 = re_sampling(method = "NeighbourhoodCleaningRule")
import warnings
warnings.filterwarnings('ignore')
fc = Feature_Scaling()
fd = Feature_Discretizer()
class BASE_MODEL_BOOSTING(object):

    def __init__(self):
        self.data = DATA()

        self.classifier_lis = [

            AdaBoostClassifier(random_state=0), \
            BaggingClassifier(random_state=0), \
            DecisionTreeClassifier(random_state=0), \
            ExtraTreesClassifier(random_state=0), \
            GradientBoostingClassifier(random_state=0), \
            GaussianNB(),
            KNeighborsClassifier(), \
            LogisticRegressionCV(random_state=0), \
            LinearDiscriminantAnalysis(), \
            MLPClassifier(max_iter=1000, random_state=0), \
            RandomForestClassifier(random_state=0), \
            QuadraticDiscriminantAnalysis(), \
            SVC(kernel='linear',random_state=0),

            ]



    def train(self):
        all_event = self.data.get_all_event()
        data_event,label = data.flatten_event(all_event)[:, :-1],data.flatten_event(all_event)[:, -1]
        data1 = fc.normalization(data_event, data_event)

        # data1,label = samp1.fit_resample(data1,label)

        test_event = self.data.get_all_event_test()
        test_data,test_label = self.data.flatten_event(test_event)[:,:-1],self.data.flatten_event(test_event)[:,-1]
        test_data = fc.normalization(test_data, data_event)

        model_lis = []
        f1_lis = []
        reca_lis = []
        prec_lis = []
        accuracy_lis = []
        Hss_lis = []
        Tss_lis = []
        for i in range(len(self.classifier_lis)):
            model = self.classifier_lis[i]
            model.fit(data1,label)
            step2 = model.predict(test_data)

            model_boosting = BOOSTING_1()
            pre1 = model_boosting.train()
            n = len(pre1)
            pre_label = []

            for j in range(n):
                if pre1[j] + step2[j] > 1:
                    pre_label.append(1)
                else:
                    pre_label.append(0)

            pre_label = fc.pre_deal(pre_label)
            f1 = f1_score(test_label, pre_label)
            reca = recall_score(test_label, pre_label)
            prec = precision_score(test_label, pre_label)
            accuracy = accuracy_score(test_label, pre_label)
            Hss = HSS(test_label, pre_label)
            Tss = TSS(test_label, pre_label)
            model_lis.append(str(self.classifier_lis[i]))
            f1_lis.append(f1)
            reca_lis.append(reca)
            prec_lis.append(prec)
            accuracy_lis.append(accuracy)
            Hss_lis.append(Hss)
            Tss_lis.append(Tss)

            print(str(i), str(model)[:10], f1, reca, prec, accuracy, Hss, Tss)
        # result = pd.DataFrame({"model": model_lis, "f1": f1_lis, "recall": reca_lis, "precision": prec_lis, "accuracy": accuracy_lis,"HSS": Hss_lis, "TSS": Tss_lis})
        # result.to_csv("EXPERIMENTS/result_val.csv", index=False)

if __name__ == '__main__':
    base_model_boosting = BASE_MODEL_BOOSTING()
    base_model_boosting.train()