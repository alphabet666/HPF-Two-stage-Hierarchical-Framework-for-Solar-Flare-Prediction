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
from GAM2 import GAM as GAM2
from GAM1 import GAM as GAM1
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
gam1 = GAM1()
pre1_1 = gam1.train()
gam2 = GAM2()
step2,label = gam2.train()

class Model_BOOSTING(object):

    def __init__(self):
        self.classifier_lis = [

            # imb_xgb(special_objective='weighted', imbalance_alpha=0.945),
            # Perceptron(fit_intercept=False, max_iter=10, tol=None, shuffle=False),
            # AdaBoostClassifier(random_state=0), \
            # BaggingClassifier(random_state=0), \
            # BaggingClassifier(random_state=0, max_samples=0.90), \
            # ExtraTreesClassifier(random_state=0, ), \
            # GradientBoostingClassifier(random_state=0), \
            # GradientBoostingClassifier(random_state=0, ccp_alpha=0.0001, subsample=0.95), \
            # RandomForestClassifier(random_state=0), \
            # KNeighborsClassifier(), \
            # LogisticRegressionCV(random_state=0, ), \
            # DecisionTreeClassifier(random_state=0, ), \
            # GaussianNB(), \
            # LinearDiscriminantAnalysis(), \
            # QuadraticDiscriminantAnalysis(), \
            # SVC(random_state=0),
            # EasyEnsembleClassifier(random_state=0, ),
            # BalancedBaggingClassifier(random_state=0, ),
            BalancedRandomForestClassifier(random_state=0, n_estimators=60, max_samples=2),
            RUSBoostClassifier(random_state=0, )

        ]

        self.classifier_lis_2 = [

            # AdaBoostClassifier(random_state=0), \
            # BaggingClassifier(random_state=0), \
            # BaggingClassifier(random_state=0,max_samples=0.90), \
            # ExtraTreesClassifier(random_state=0, ), \
            # GradientBoostingClassifier(random_state=0), \
            # GradientBoostingClassifier(random_state=0,ccp_alpha=0.0001,subsample=0.95), \
            # RandomForestClassifier(random_state=0), \
            # KNeighborsClassifier(), \
            # LogisticRegressionCV(random_state=0, ), \
            # DecisionTreeClassifier(random_state=0, ), \
            GaussianNB(), \
            # LinearDiscriminantAnalysis(), \
            # QuadraticDiscriminantAnalysis(), \
            # SVC(random_state=0),
            # EasyEnsembleClassifier(random_state=0, ),
            # BalancedBaggingClassifier(random_state=0, ),
            # BalancedRandomForestClassifier(random_state=0, n_estimators=60, max_samples=2),
            # RUSBoostClassifier(random_state=0, )
        ]



        event = data.step1_data_train()
        self.data_all1, self.label1 = event[:, :-1], event[:, -1]
        self.data_all2, self.label2 = samp1.fit_resample(event[:, :-1], event[:, -1])

        self.conf_dict = None

    def update_settings(self, conf_dict):
            self.conf_dict = conf_dict

    def train(self):
        for k in range(len(self.classifier_lis)):  # 使用不同的分类器
            # 第一阶段分类器
            conf_dict = {}
            conf_dict['CLASSIFIER_1_ID'] = k
            for j in range(len(self.classifier_lis_2)):
                conf_dict['CLASSIFIER_2_ID'] = j

                model_1 = self.classifier_lis[conf_dict['CLASSIFIER_1_ID']]

                model_1.fit(self.data_all1, self.label1)

                model_2 = self.classifier_lis_2[conf_dict['CLASSIFIER_2_ID']]
                model_2.fit(self.data_all1, self.label1)

                test_event = data.step1_data_test()
                test_data, test_label = test_event[:, :-1], test_event[:, -1]

                pre1_2 = model_1.predict(test_data)
                pre1_3 = model_2.predict(test_data)

                n = len(pre1_1)
                step1 = []
                for i in range(n):
                    if pre1_1[i] > 0 and pre1_2[i] > 0 and pre1_3[i] > 0:
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
                prec, f1, reca, accuracy = self.metric(pre_label, label)
                conf_dict['metric'] = [prec, f1, reca, accuracy]

                print(self.metric(step1, test_label))
                print(conf_dict)


    def metric(self,pre,test_label):
        pre = fc.pre_deal(pre)
        f1 = f1_score(test_label, pre)
        reca = recall_score(test_label, pre)
        prec = precision_score(test_label, pre)
        accuracy = accuracy_score(test_label, pre)
        return f1, reca,prec,accuracy



if __name__ == '__main__':

    model_boosting = Model_BOOSTING()
    model_boosting.train()






