import numpy as np
from Data import DATA
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,roc_auc_score,fbeta_score
from Metric import HSS,TSS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.ensemble import RUSBoostClassifier
from Feature_Processing import Feature_Scaling,Feature_Discretizer
from Feature_Processing import Feature_Scaling
data = DATA()
fc = Feature_Scaling()
from step1 import Model_BOOSTING as BOOSTING_1
from pygam import LogisticGAM
from sampling_for_imbalance import re_sampling  #采样
samp1 = re_sampling(method = "NeighbourhoodCleaningRule")
import warnings
warnings.filterwarnings('ignore')
fc = Feature_Scaling()
fd = Feature_Discretizer()
class BASE_MODEL_BOOSTING(object):

    def __init__(self):
        self.data = DATA()


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

        test_data_s = fc.standardization(test_data, data_event)
        test_data_n = fc.normalization(test_data, data_event)
        x_1 = fc.standardization(x_1, data_event)

        constraints = [None, None, "concave", None, "concave", "concave", "concave", "concave", "concave", None]
        model_1 = LogisticGAM(constraints=constraints).gridsearch(x_1, y_1)
        model_2 = RUSBoostClassifier(random_state=0)
        model_3 = LinearDiscriminantAnalysis()

        model_1.fit(x_1, y_1)
        model_2.fit(x_1, y_1)
        model_3.fit(x_1, y_1)
        pre1_1 = model_1.predict(test_data_s)
        pre1_2 = model_1.predict(test_data_s)
        pre1_3 = model_1.predict(test_data_s)

        data_event = data.flatten_event(all_event)[:, :-1]

        data1 = fc.normalization(data_event, data_event)
        label = data.flatten_event(all_event)[:, -1]
        model = LinearDiscriminantAnalysis()
        model.fit(data1,label)
        pre = model.predict_proba(test_data_n)[:,1]

        n = len(pre1_1)
        step2 = []
        for i in range(n):
            if pre1_1[i] + pre1_2[i] + pre1_3[i] > 0:
                step2.append(1)
            else:
                step2.append(0)


        model_boosting = BOOSTING_1()
        pre1 = model_boosting.train()
        n = len(pre1)
        pre_label = []

        for j in range(n):
            if pre1[j] > 0 and step2[j] > 0:
                pre_label.append(1)
            elif pre[j] > 0.5:
                pre_label.append(1)
            else:
                pre_label.append(0)

        pre_label = fc.pre_deal(pre_label)
        f1 = f1_score(test_label, pre_label)
        f05 = fbeta_score(test_label, pre_label, beta=0.5)
        reca = recall_score(test_label, pre_label)
        prec = precision_score(test_label, pre_label)
        accuracy = accuracy_score(test_label, pre_label)
        # Hss = HSS(test_label, pre_label)
        # Tss = TSS(test_label, pre_label)
        # AUC = roc_auc_score(test_label, pre_label)

        print("----------")
        print( f1, f05,reca, prec, accuracy)
        # result = pd.DataFrame({"model": model_lis, "f1": f1_lis, "recall": reca_lis, "precision": prec_lis, "accuracy": accuracy_lis,"HSS": Hss_lis, "TSS": Tss_lis})
        # result.to_csv("EXPERIMENTS/result_val.csv", index=False)

if __name__ == '__main__':
    base_model_boosting = BASE_MODEL_BOOSTING()
    base_model_boosting.train()