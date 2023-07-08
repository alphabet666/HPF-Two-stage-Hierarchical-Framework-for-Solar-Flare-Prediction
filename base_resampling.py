from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC,OneClassSVM
from sklearn.neural_network import MLPClassifier
from Data import DATA
from imblearn.over_sampling import SMOTE
from Feature_Processing import Feature_Scaling,Feature_Discretizer
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,roc_auc_score,fbeta_score
from Metric import HSS,TSS
import pandas as pd
from sampling_for_imbalance import re_sampling  #采样
import joblib
import os
import numpy as np
from imxgboost.imbalance_xgb import imbalance_xgboost as imb_xgb
#分类器集成
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import RUSBoostClassifier
import csv
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"


classifier_lis = [
    # DecisionTreeClassifier(), \
    # LogisticRegressionCV(), \
    # LinearDiscriminantAnalysis(), \
    # QuadraticDiscriminantAnalysis(), \
    # SVC(),
    # GaussianNB(),
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
    SVC(random_state=0)

    # RUSBoostClassifier(random_state=0),
    # EasyEnsembleClassifier(random_state=0),
    # BalancedBaggingClassifier(random_state=0),
    # BalancedRandomForestClassifier(random_state=0),

    ]

def get_all_event(PATH_1):

    event_lis = []
    file_lis = os.listdir(PATH_1)
    for f in file_lis:
        file_path = PATH_1 + f
        npy_file = np.load(file_path)
        n = npy_file.shape[0]
        if n > 1:
            event_lis.append(npy_file.copy())
    event_lis = np.array(event_lis)

    return event_lis

def get_all_event_test(PATH_test_1):

    event_lis = []
    DIR_PATH = PATH_test_1
    file_lis = os.listdir(DIR_PATH)
    for f in file_lis:
        file_path = DIR_PATH + f
        npy_file = np.load(file_path)
        event_lis.append(npy_file.copy())
    event_lis = np.array(event_lis)
    return event_lis

methods = ["RandomOverSampler","SMOTE","ADASYN","RandomUnderSampler","NeighbourhoodCleaningRule","EditedNearestNeighbours"]

path_trains = ['./data/All/partition/NPY_DATA/','./data/All/partition2/NPY_DATA/', \
    './data/All/partition3/NPY_DATA/','./data/All/partition4/NPY_DATA/', \
        './data/All/partition5/NPY_DATA/','./data/All/partition6/NPY_DATA/','./data/All/partition7/NPY_DATA/']

path_tests = ["./data/All/partition/NPY_test/","./data/All/partition2/NPY_test/","./data/All/partition3/NPY_test/",\
    "./data/All/partition4/NPY_test/","./data/All/partition5/NPY_test/","./data/All/partition6/NPY_test/",\
        "./data/All/partition7/NPY_test/"]

for method_ in methods:
    samp = re_sampling(method = method_)
    csv1 = []
    for j in range(len(path_trains)):

        data = DATA()
        fd = Feature_Discretizer()
        fc = Feature_Scaling()

        all_event = get_all_event(path_trains[j])
        data_event = data.flatten_event(all_event)[:,:-1]
        data1 = fc.normalization(data_event,data_event) #标准化
        label = data.flatten_event(all_event)[:, -1] #最后一列作为标签

        X_smo, y_smo = data1, label
        X_smo, y_smo = samp.fit_resample(data1, label)

        #获取验证集
        val_event = data.get_all_event_val()
        val_data = fc.normalization(data.flatten_event(val_event)[:,:-1],data_event)
        val_label = data.flatten_event(val_event)[:, -1]

        test_event = get_all_event_test(path_tests[j])
        test_data = fc.normalization(data.flatten_event(test_event)[:,:-1],data_event)
        test_label = data.flatten_event(test_event)[:, -1]

        
        model_lis = []
        f1_lis = []
        reca_lis = []
        prec_lis = []
        accuracy_lis = []
        Hss_lis = []
        Tss_lis = []
        bagging_pre = []
        n = len(test_label)
        for i in range(len(classifier_lis)):
            model = classifier_lis[i]
            model.fit(X_smo, y_smo)
            pre = model.predict(test_data)
            pre = fc.pre_deal(pre)
            f1 = f1_score(test_label, pre)
            f05 = fbeta_score(test_label,pre,beta=0.5)
            #reca = recall_score(test_label, pre)
            #prec = precision_score(test_label, pre)
            #accuracy = accuracy_score(test_label, pre)
            # Hss = HSS(test_label, pre)
            # Tss = TSS(test_label, pre)
            # model_lis.append(str(classifier_lis[i]))
            # f1_lis.append(f1)
            # reca_lis.append(reca)
            # prec_lis.append(prec)
            # accuracy_lis.append(accuracy)
            # Hss_lis.append(Hss)
            # Tss_lis.append(Tss)
            print(str(classifier_lis[i])[:10],str(path_trains[j]),f05,f1)
            a = [str(classifier_lis[i])[:10],str(path_trains[j]),f05,f1]
            csv1.append(a)

    csvname = method_+'.csv'
    with open(csvname, 'w', newline='') as file:
        writer = csv.writer(file)

        # 写入头部（可选）
        writer.writerow(['Model', 'partition','F05', 'F1'])

        # 逐行写入数据
        for row in csv1:
            writer.writerow(row)

    print(csvname + "has been done!")

    # 集成预测
#     if f1 > 0.4:
#         pre = model.predict(test_data)
#         bagging_pre.append(pre)
# bagging_pre = np.array(bagging_pre)
# pre_mean = np.mean(bagging_pre,axis=0)
# pre = []
# for x in pre_mean:
#     if x > 0.5:
#         pre.append(1)
#     else:
#         pre.append(0)
# f1 = f1_score(test_label, pre)
# print(f1,recall_score(test_label, pre),precision_score(test_label, pre),accuracy_score(test_label, pre),HSS(test_label, pre),TSS(test_label, pre))
#


#     if f1 > 0.1:
#         joblib.dump(model, os.getcwd() + '/saved_conf/baseline/' +str(i) +"_"+ str(f1)[0:9] + "-" + ".model")
# result = pd.DataFrame({"model": model_lis, "f1": f1_lis, "recall": reca_lis, "precision": prec_lis, "accuracy": accuracy_lis,"HSS": Hss_lis, "TSS": Tss_lis})
# result.to_csv("EXPERIMENTS/baseline/val-23.csv", index=False)

