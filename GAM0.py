from Data import DATA
from Feature_Processing import Feature_Scaling,Feature_Discretizer
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,fbeta_score
from Metric import HSS,TSS
from pygam import LogisticGAM

import warnings
warnings.filterwarnings('ignore')
data = DATA()
fd = Feature_Discretizer()
fc = Feature_Scaling()
all_event = data.get_all_event()

data_event = data.flatten_event(all_event)[:,:-1]

data1 = fc.normalization(data_event,data_event)
label = data.flatten_event(all_event)[:, -1]


val_event = data.get_all_event_test()
val_data = fc.normalization(data.flatten_event(val_event)[:,:-1],data_event)
val_label = data.flatten_event(val_event)[:, -1]

test_event = data.get_all_event_test()
test_data = fc.normalization(data.flatten_event(test_event)[:,:-1],data_event)
test_label = data.flatten_event(test_event)[:, -1]

model_lis = []
f1_lis = []
f05_lis = []
reca_lis = []
prec_lis = []
accuracy_lis = []
Hss_lis = []
Tss_lis = []
bagging_pre = []
n = len(test_label)
# constraints = [None,None,"concave",None,"concave","concave","concave","concave","concave",None]
constraints = ["concave","concave","concave","concave","concave","concave","concave","concave","concave","concave"]
model = LogisticGAM(constraints=constraints).gridsearch(data1, label)

pre = model.predict(val_data)
pre = fc.pre_deal(pre)
f1 = f1_score(val_label, pre)
f05 = fbeta_score(val_label, pre, beta=0.5)
reca = recall_score(val_label, pre)
prec = precision_score(val_label, pre)
accuracy = accuracy_score(val_label, pre)
Hss = HSS(val_label, pre)
Tss = TSS(val_label, pre)

f1_lis.append(f1)
f05_lis.append(f05)
reca_lis.append(reca)
prec_lis.append(prec)
accuracy_lis.append(accuracy)
Hss_lis.append(Hss)
Tss_lis.append(Tss)
print(f1,f05,reca,prec,accuracy,Hss,Tss)


