import numpy as np
from sklearn.metrics import confusion_matrix

def prep_clf(obs,pre):
    '''
    func: 计算二分类结果-混淆矩阵的四个元素
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；

    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN
    '''
    # print("obs.shape,pre.shape",obs.shape,pre.shape)
    cm = confusion_matrix(obs,pre)
    cm = cm.astype(np.float32)
    #print("cm",cm)
    FP = cm[0,1]
    FN = cm[1,0]
    TN = cm[0,0]
    TP = cm[1,1]

    #print("obs.shape,pre.shape",obs,pre)
    #TN,FP,FN,TP = confusion_matrix(obs,pre).ravel
    #cm = cm.astype(np.float32)
    # print("cm",cm)
    # FP = cm[0,1]
    # FN = cm[1,0]
    # TN = cm[0,0]
    # TP = cm[1,1]

    # TP, FN, FP, TN = compute_confusion_matrix(obs,pre)

    return TP, FN, FP, TN

def HSS(obs, pre):
    '''
    HSS - Heidke skill score
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): pre

    Returns:
        float: HSS value
    '''
    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre)
    #print("TP, FN, FP, TN",TP, FN, FP, TN)
    HSS_num = 2 * (TP * TN - FN * FP)
    HSS_den = (TP+FN)*(TN+FN)+(TP+FP)*(TN+FP)
    #print("HSS_den",HSS_den)
    return HSS_num / HSS_den

def TSS(obs, pre):
    '''
    TSS - The True Skill Statistic
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): pre

    Returns:
        float: TSS value

    other:
    FAR : false alarm rate
    '''
    TP, FN, FP, TN = prep_clf(obs=obs, pre=pre)
    recall = TP/(TP+FN)
    FAR = FP/(FP+TN)

    return recall - FAR

def compute_confusion_matrix(y_true, y_pred):
    TP = 0   # 真正例 True Positive
    TN = 0   # 真负例 True Negative
    FP = 0   # 假正例 False Positive
    FN = 0   # 假负例 False Negative

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1

    return TP, FN, FP, TN

