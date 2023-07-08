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
from sklearn.neural_network import MLPClassifier
from GAM2 import GAM as GAM2
from GAM1 import GAM as GAM1
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from sampling_for_imbalance import re_sampling
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,fbeta_score
from Feature_Processing import Feature_Scaling,Feature_Discretizer
import warnings
from Metric import HSS,TSS
import csv
warnings.filterwarnings('ignore')
fc = Feature_Scaling()
fd = Feature_Discretizer()
data = DATA()
samp1 = re_sampling(method = "NeighbourhoodCleaningRule")
gam1 = GAM1()
pre1_1 = gam1.train()
gam2 = GAM2()
step2,label = gam2.train()

class Model_BOOSTING(object):

    def __init__(self):
        self.classifier_lis = [

            # imb_xgb(special_objective='weighted', imbalance_alpha=0.945),
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


        event = data.step1_data_train(data_type = "new_feature")
        self.data_all1, self.label1 = event[:, :-1], event[:, -1]
        self.data_all2, self.label2 = samp1.fit_resample(event[:, :-1], event[:, -1])

        self.conf_dict = None

    def update_settings(self, conf_dict):
            self.conf_dict = conf_dict

    def train(self):
        csv1 = []
        for k in range(len(self.classifier_lis)):  # 使用不同的分类器
            # 第一阶段分类器
            conf_dict = {}
            conf_dict['CLASSIFIER_1_ID'] = k
            
            model_1 = self.classifier_lis[conf_dict['CLASSIFIER_1_ID']]

            model_1.fit(self.data_all2, self.label2)

            test_event = data.step1_data_test(data_type = "new_feature")
            test_data, test_label = test_event[:, :-1], test_event[:, -1]

            pre1_2 = model_1.predict(test_data)

            n = len(pre1_1)
            step1 = []
            for i in range(n):
                if pre1_1[i] > 0 and pre1_2[i] > 0:
                    step1.append(1)
                else:
                    step1.append(0)

            n = len(pre1_1)
            pre_label = []
            for j in range(n):
                if step1[j] > 0 and step2[j] > 0:
                    pre_label.append(1)
                else:
                    pre_label.append(0)
            f1, f05,  reca,prec, accuracy = self.metric(pre_label, label)
            Hss = HSS(pre_label, label)
            Tss = TSS(pre_label, label)
            #conf_dict['metric'] = [f1, f05, reca, prec,accuracy]
            # if prec > 0.1 and f1 > 0.5:
            #     print("f1, f05, reca, prec, accuracy")
            #     print(self.metric(step1, test_label))
            #     print(conf_dict)
            a = [str(model_1)[:10],f1,f05,reca,prec,accuracy, Hss, Tss]
            #print("f1, f05, reca, prec, accuracy")
            print(str(model_1)[:10], f1, f05,reca, prec, accuracy, Hss, Tss)
            #print(conf_dict)
            csv1.append(a)

        with open('test_HPF_base_output_partition7.csv', 'w', newline='') as file:
            writer = csv.writer(file)

    # 写入头部（可选）
            writer.writerow(['Model', 'F1', 'F0.5','Recall','Precision',"accuracy","Hss","Tss"])

    # 逐行写入数据
            for row in csv1:
                writer.writerow(row)

        print("Data has been written to test_HPF_base_output_partition7.csv")


    def metric(self,pre,test_label):
        pre = fc.pre_deal(pre)
        f1 = f1_score(test_label, pre)
        f05 = fbeta_score(test_label, pre, beta=0.5)
        reca = recall_score(test_label, pre)
        prec = precision_score(test_label, pre)
        accuracy = accuracy_score(test_label, pre)
        return f1, f05,reca,prec,accuracy



if __name__ == '__main__':

    model_boosting = Model_BOOSTING()
    model_boosting.train()






