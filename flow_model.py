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
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
from sampling_for_imbalance import re_sampling
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
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

        self.classifier_lis_1 = [
            # imb_xgb(special_objective='weighted', imbalance_alpha=0.5),
            # Perceptron(fit_intercept=False, max_iter=50, tol=None, shuffle=False),
            AdaBoostClassifier(random_state=0), \
            BaggingClassifier(random_state=0), \
            BaggingClassifier(n_estimators= 50,random_state=0,max_samples=0.90), \
            ExtraTreesClassifier(random_state=0, ), \
            GradientBoostingClassifier(random_state=0), \
            GradientBoostingClassifier(random_state=0,ccp_alpha=0.0001,subsample=0.95), \
            RandomForestClassifier(random_state=0), \
            KNeighborsClassifier(), \
            LogisticRegressionCV(random_state=0, ), \
            DecisionTreeClassifier(random_state=0, ), \
            GaussianNB(var_smoothing=0.25), \
            LinearDiscriminantAnalysis(), \
            QuadraticDiscriminantAnalysis(), \
            SVC(kernel='linear',random_state=0,probability=True),
            EasyEnsembleClassifier(random_state=0, ),
            BalancedBaggingClassifier(random_state=0, ),
            BalancedRandomForestClassifier(random_state=0, ),
            RUSBoostClassifier(random_state=0, )
        ]

        self.classifier_lis_2 = [
            # imb_xgb(special_objective='focal', focal_gamma=1.5),
            # Perceptron(fit_intercept=False, max_iter=10, tol=None, shuffle=False),
            # AdaBoostClassifier(random_state=0), \
            BaggingClassifier(random_state=0), \
            BaggingClassifier(n_estimators= 50,random_state=0,max_samples=0.90), \
            # ExtraTreesClassifier(random_state=0, ), \
            # GradientBoostingClassifier(random_state=0), \
            # GradientBoostingClassifier(random_state=0,ccp_alpha=0.0001,subsample=0.95), \
            # RandomForestClassifier(random_state=0), \
            # KNeighborsClassifier(), \
            # LogisticRegressionCV(random_state=0, ), \
            # DecisionTreeClassifier(random_state=0, ), \
            # GaussianNB(var_smoothing=0.25), \
            # LinearDiscriminantAnalysis(), \
            # QuadraticDiscriminantAnalysis(), \
            # SVC(kernel='linear',random_state=0,probability=True),
            # EasyEnsembleClassifier(random_state=0, ),
            # BalancedBaggingClassifier(random_state=0, ),
            # BalancedRandomForestClassifier(random_state=0, ),
            # RUSBoostClassifier(random_state=0, )
        ]
        event = data.step1_data_train()
        self.data_all1, self.label1 = event[:, :-1], event[:, -1]
        self.data_all2, self.label2 = samp1.fit_resample(event[:, :-1], event[:, -1])

        # max_event = data.step1_data_train(data_type="max")
        # self.x1, self.y1 = max_event[:, :-1], max_event[:, -1]
        # self.x1, self.y1 = samp2.fit_resample(self.x1, self.y1)
        #
        # mean_event = data.step1_data_train(data_type="mean")
        # self.x2, self.y2 = mean_event[:, :-1], mean_event[:, -1]
        # self.x2, self.y2 = samp2.fit_resample( self.x2, self.y2 )


        self.conf_dict = None

    def update_settings(self, conf_dict):
            self.conf_dict = conf_dict

    def train(self):
        model_1_1 = imb_xgb(special_objective='weighted', imbalance_alpha=0.945)
        model_1_2 = BalancedRandomForestClassifier(random_state=0, n_estimators=60, max_samples=2)
        model_1_1.fit(self.data_all1, self.label1)
        model_1_2.fit(self.data_all2, self.label2)

        for k in range(len(self.classifier_lis_1)):  # 使用不同的分类器
            # 第一阶段分类器
            conf_dict = {}
            conf_dict['CLASSIFIER_1_ID'] = k
            for j in range(len(self.classifier_lis_2)):

                conf_dict['CLASSIFIER_2_ID'] = j

                model_2 = self.classifier_lis_1[conf_dict['CLASSIFIER_1_ID']]
                model_2.fit(self.x1, self.y1 )

                model_3 = self.classifier_lis_2[conf_dict['CLASSIFIER_2_ID']]
                model_3.fit(self.x2, self.y2)

                val_event = data.step1_data_test()
                val_data, val_label = val_event[:, :-1], val_event[:, -1]

                val_max = data.step1_data_test(data_type="max")
                val_x1, val_y1 = val_max[:, :-1], val_max[:, -1]

                val_mean = data.step1_data_test(data_type="mean")
                val_x2, val_y2 = val_mean[:, :-1], val_mean[:, -1]

                pre1_1_1 = model_1_1.predict_sigmoid(val_data)
                pre1_1_2 = model_1_2.predict_proba(val_data)[:,1]
                pre1_2 = model_2.predict_proba(val_x1)[:,1]
                pre1_3 = model_3.predict_proba(val_x2)[:,1]


                # print(self.metric(pre1_1, val_label))
                # print(self.metric(pre1_2, val_label))
                # print(self.metric(pre1_3, val_label))

                n = len(pre1_1_1)
                pre1 = []
                for i in range(n):
                    if (pre1_1_1[i] > 0.55 and pre1_1_2[i] > 0.55) or  pre1_2[i] > np.percentile(pre1_2,99.5) or  pre1_3[i] > np.percentile(pre1_3,99.5):
                        pre1.append(1)
                    else:
                        pre1.append(0)
                f1,reca,prec,accuracy = self.metric(pre1, val_label)
                conf_dict['metric'] = [f1, reca,prec,accuracy]
                if prec > 0.1 and reca > 0.1:
                    print(conf_dict)
                    # print(self.metric(pre1_1, val_label))
                    # print(self.metric(pre1_2, val_label))
                    # print(self.metric(pre1_3, val_label))




    def metric(self,pre,test_label):
        pre = fc.pre_deal(pre)
        f1 = f1_score(test_label, pre)
        reca = recall_score(test_label, pre)
        prec = precision_score(test_label, pre)
        accuracy = accuracy_score(test_label, pre)
        return f1,reca,prec,accuracy



if __name__ == '__main__':

    model_boosting = Model_BOOSTING()
    model_boosting.train()






