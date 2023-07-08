
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
from sklearn.model_selection import GridSearchCV
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
samp2 = re_sampling(method = "KMeansSMOTE")

class Model_BOOSTING(object):

    def __init__(self):
        self.classifier_lis = [

            # imb_xgb(special_objective='weighted', imbalance_alpha=0.945),
            # imb_xgb(special_objective='weighted'),
            # Perceptron(fit_intercept=False, max_iter=10, tol=None, shuffle=False),
            # MLPClassifier(hidden_layer_sizes=(100,)),
            AdaBoostClassifier(random_state=0), \
            # BaggingClassifier(random_state=0), \
            # BaggingClassifier(random_state=0, max_samples=0.95), \
            # ExtraTreesClassifier(random_state=0, ), \
            GradientBoostingClassifier(random_state=0), \
            # GradientBoostingClassifier(random_state=0, ccp_alpha=0.0001, subsample=0.95), \
            # RandomForestClassifier(random_state=0), \
            # KNeighborsClassifier(), \
            # LogisticRegressionCV(random_state=0), \
            # DecisionTreeClassifier(random_state=0, ), \
            GaussianNB(), \
            # LinearDiscriminantAnalysis(), \
            # QuadraticDiscriminantAnalysis(), \
            # QuadraticDiscriminantAnalysis(priors=[0.05,0.95]), \
            #
            # QuadraticDiscriminantAnalysis(priors=[0.95,0.05]), \
            #
            # SVC(),
            # EasyEnsembleClassifier(random_state=0, ),
            # BalancedBaggingClassifier(random_state=0, ),
            # BalancedRandomForestClassifier(random_state=0),
            BalancedRandomForestClassifier(random_state=0, n_estimators=60,max_samples=2 ),
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

            model_1 = self.classifier_lis[conf_dict['CLASSIFIER_1_ID']]
            model_1.fit(self.data_all2, self.label2)



            val_event = data.step1_data_test()
            val_data, val_label = val_event[:, :-1], val_event[:, -1]

            pre1_1 = model_1.predict(val_data)


            # n = len(pre1_1)
            # pre1 = []
            # for i in range(n):
            #     if pre1_1[i] > 0:
            #         pre1.append(1)
            #     else:
            #         pre1.append(0)
            prec, f1, reca, accuracy = self.metric(pre1_1, val_label)
            conf_dict['metric'] = [prec, f1, reca, accuracy]
            if prec > 0.1 and f1 > 0.1:
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






