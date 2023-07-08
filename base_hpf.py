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
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,fbeta_score
from Metric import HSS,TSS
from Feature_Processing import Feature_Scaling,Feature_Discretizer
from sklearn.neural_network import MLPClassifier
from Feature_Processing import Feature_Scaling
import csv
data = DATA()
fc = Feature_Scaling()

#from base_model_step2 import BASE_MODEL_BOOSTING as BOOSTING_2
from step1 import Model_BOOSTING as BOOSTING_1
from sklearn.linear_model import Perceptron
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier
import pandas as pd
from pygam import LogisticGAM
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from sampling_for_imbalance import re_sampling  #采样
samp1 = re_sampling(method = "NeighbourhoodCleaningRule")
import warnings
warnings.filterwarnings('ignore')
fc = Feature_Scaling()
fd = Feature_Discretizer()
class BASE_MODEL_BOOSTING(object):

    def __init__(self):
        self.data = DATA()

        constraints = [None,None,"concave",None,"concave","concave","concave","concave","concave",None]

        self.classifier_lis = [
            # imb_xgb(special_objective='focal', focal_gamma= 2),
            #LogisticGAM(constraints=constraints),
            # AdaBoostClassifier(n_estimators=300,learning_rate=0.9), 
            # BaggingClassifier(random_state=0), \
            # DecisionTreeClassifier(random_state=0), \
            # ExtraTreesClassifier(random_state=0), \
            # GradientBoostingClassifier(random_state=0), \
            # GaussianNB(),
            # KNeighborsClassifier(), \
            # LogisticRegressionCV(random_state=0), \
            # LinearDiscriminantAnalysis(), \
            # MLPClassifier(random_state=0), \
            # RandomForestClassifier(random_state=0, n_estimators=30,ccp_alpha=0.00015),
            # QuadraticDiscriminantAnalysis(), \
            # SVC(kernel='linear',random_state=0,probability=True),

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
            SVC(random_state=0), \
            ]

    def get_data_1(self,all,mean):
        which_1 = np.where(mean[:, -1] == 1)
        y_1 = self.data.flatten_event(all)[which_1][:, -1]
        x_1 = self.data.flatten_event(all)[which_1][:, :-1]
        return x_1,y_1

    def train(self):
        all_event = self.data.get_all_event()
        data_event = data.flatten_event(all_event)[:, :-1]
        new_label = self.data.flatten_event(self.data.new_label_event())
        x_1, y_1 = self.get_data_1(all_event, new_label)

        test_event = self.data.get_all_event_test()
        test_data, test_label = self.data.flatten_event(test_event)[:, :-1], self.data.flatten_event(test_event)[:, -1]

        test_data = fc.standardization(test_data, data_event)
        x_1 = fc.standardization(x_1, data_event)
        csv1 = []

        for i in range(len(self.classifier_lis)):
            model = self.classifier_lis[i]
            model.fit(x_1, y_1)
            #step2 = model.predict_proba(test_data)
            step2 = model.predict(test_data)
            step2 = fc.pre_deal(step2)
            
            model_boosting = BOOSTING_1()
            pre1 = model_boosting.train()
            n = len(pre1)
            pre_label = []

            for j in range(n):
                if pre1[j] > 0 and step2[j] > 0:
                    pre_label.append(1)
                else:
                    pre_label.append(0)

            pre_label = fc.pre_deal(pre_label)
            f1 = f1_score(test_label, pre_label)
            f05 = fbeta_score(test_label, pre_label,beta=0.5)
            reca = recall_score(test_label, pre_label)
            prec = precision_score(test_label, pre_label)
            accuracy = accuracy_score(test_label, pre_label)
            # Hss = HSS(test_label, pre_label)
            # Tss = TSS(test_label, pre_label)

            #print(str(i), str(model)[:10], f1, reca, prec, accuracy, Hss, Tss)
            print(str(i), str(model)[:10], f1, f05,reca, prec, accuracy)
            a = [str(model)[:10],f1,f05,reca,prec,accuracy]
            csv1.append(a)

        with open('HPF_base_output_partition8.csv', 'w', newline='') as file:
            writer = csv.writer(file)

    # 写入头部（可选）
            writer.writerow(['Model', 'F1', 'F0.5','Recall','Precision',"accuracy"])

    # 逐行写入数据
            for row in csv1:
                writer.writerow(row)

        print("Data has been written to HPF_base_output_partition8.csv")
        # result = pd.DataFrame({"model": model_lis, "f1": f1_lis, "recall": reca_lis, "precision": prec_lis, "accuracy": accuracy_lis,"HSS": Hss_lis, "TSS": Tss_lis})
        # result.to_csv("EXPERIMENTS/result_val.csv", index=False)

if __name__ == '__main__':
    base_model_boosting = BASE_MODEL_BOOSTING()
    base_model_boosting.train()