
import numpy as np
from Data import DATA
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,fbeta_score
from Metric import HSS,TSS
from Feature_Processing import Feature_Scaling,Feature_Discretizer
from pygam import LogisticGAM

import warnings
warnings.filterwarnings('ignore')
fc = Feature_Scaling()
fd = Feature_Discretizer()

class GAM(object):

    def __init__(self):
        self.data = DATA()

    def get_data_1(self,all,mean):
        which_1 = np.where(mean[:, -1] == 1)
        y_1 = self.data.flatten_event(all)[which_1][:, -1]
        x_1 = self.data.flatten_event(all)[which_1][:, :-1]
        return x_1,y_1

    def train(self):
        all_event = self.data.get_all_event()
        all_data = self.data.flatten_event(all_event)[:, :-1]
        new_label = self.data.flatten_event(self.data.new_label_event())
        x_1, y_1 = self.get_data_1(all_event,new_label)


        val_event = self.data.get_all_event_val()
        new_label_val = self.data.flatten_event(self.data.new_label_val())
        val_data, val_label = self.get_data_1(val_event,new_label_val)

        val_data = fc.standardization(val_data,all_data)
    
        test_event = self.data.get_all_event_test()
        new_label_test = self.data.flatten_event(self.data.new_label_test())
        test_data, test_label = self.data.flatten_event(test_event)[:,:-1],self.data.flatten_event(test_event)[:,-1]
        test_data = fc.standardization(test_data,all_data)

        x_1 = fc.standardization(x_1,all_data)

        constraints = ["concave"] * 10


        # constraints = [None,None,"concave",None,"concave","concave","concave","concave","concave",None]
        # constraints = ["convex", "concave", "monotonic_inc", "monotonic_dec","circular",  None, None, None, None]

        model = LogisticGAM(constraints=constraints).gridsearch(x_1, y_1)

        model_lis = []
        f1_lis = []
        f05_lis = []
        reca_lis = []
        prec_lis = []
        accuracy_lis = []
        Hss_lis = []
        Tss_lis = []
        bagging_pre =[]
        pre = model.predict(val_data)
        f1 = f1_score(val_label, pre)
        f05 = fbeta_score(val_label, pre, beta=0.5)
        reca = recall_score(val_label, pre)
        prec = precision_score(val_label, pre)
        accuracy = accuracy_score(val_label, pre)
        Hss = HSS(val_label, pre)
        Tss = TSS(val_label, pre)
        f1_lis.append(f1)
        reca_lis.append(reca)
        prec_lis.append(prec)
        accuracy_lis.append(accuracy)
        Hss_lis.append(Hss)
        Tss_lis.append(Tss)

        print(str(model)[:10],f1,f05,reca,prec,accuracy,Hss,Tss)

        return model.predict(test_data),test_label



if __name__ == '__main__':
    base_model_boosting = GAM()
    base_model_boosting.train()