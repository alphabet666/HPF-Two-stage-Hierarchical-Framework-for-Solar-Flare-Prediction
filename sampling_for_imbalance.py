import numpy as np
#上采样
from imblearn.over_sampling import SMOTE, ADASYN  #SMOTE和ADASYN通过插值产生新的样本
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import RandomOverSampler #采用随机过采样的方式

#下采样
from imblearn.under_sampling import RandomUnderSampler #是一种快速和简单的方法来平衡数据，随机选择一个子集的数据为目标类，且可以对异常数据进行处理
from imblearn.under_sampling import ClusterCentroids #利用K-means来减少样本的数量。因此，每个类的合成都将以K-means方法的中心点来代替原始样本。
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import InstanceHardnessThreshold

#上采样和下采样的融合
from imblearn.combine import SMOTEENN,SMOTETomek



method_lis = [
    SMOTE(random_state = 1000), \
    ADASYN(random_state = 42), \
    BorderlineSMOTE(random_state = 42), \
    SVMSMOTE(random_state = 42), \
    KMeansSMOTE(random_state = 42), \
    SMOTENC(categorical_features=[1,0]), \
    RandomOverSampler(random_state = 42), \

    RandomUnderSampler(random_state = 42), \
    ClusterCentroids(random_state = 42), \
    EditedNearestNeighbours(), \
    RepeatedEditedNearestNeighbours(), \
    AllKNN(),
    CondensedNearestNeighbour(random_state = 42), \
    OneSidedSelection(random_state = 42), \
    NeighbourhoodCleaningRule(), \
    InstanceHardnessThreshold(random_state = 42),

    SMOTEENN(random_state = 42),
    SMOTETomek(random_state = 42)
]



method_id = {"SMOTE":0,"ADASYN":1,"BorderlineSMOTE":2,"SVMSMOTE":3,"KMeansSMOTE":4,"SMOTENC":5,"RandomOverSampler":6,"RandomUnderSampler":7,
                       "ClusterCentroids":8,"EditedNearestNeighbours":9,"RepeatedEditedNearestNeighbours":10,"AllKNN":11,"CondensedNearestNeighbour":12,
          "OneSidedSelection":13,"NeighbourhoodCleaningRule":14,"InstanceHardnessThreshold":15,"SMOTEENN":16,"SMOTETomek":17}



def re_sampling(method = "SMOTE"):

    method_model = method_lis[method_id[method]]

    return method_model
