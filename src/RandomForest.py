import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
import pandas as pd
import time
import warnings
from sklearn import metrics
from sklearn.model_selection import KFold
# from sklearn import tree
# from sklearn.externals.six import StringIO
# import pydot
# from IPython.display import Image
warnings.filterwarnings("ignore", category=DeprecationWarning)


def classifyRes(arr):
    arr = arr.reshape(arr.size,)
    for i in range(arr.size):
        arr[i] = 1 if arr[i]>0.5 else 0
    return arr

def calMetrix(y_pre, test_y):
    print('confusion_matrix:')
    confm = metrics.confusion_matrix(y_pre, test_y)
    print(confm)
    print('accuracy_score: ')
    print(metrics.accuracy_score(y_pre, test_y))
    print('precision_score: ')
    print(metrics.precision_score(y_pre, test_y))
    print('recall_score: ')
    print(metrics.recall_score(y_pre, test_y, average='binary'))
    print('f1_score: ')
    print(metrics.f1_score(y_pre, test_y))
    TN, FP, FN, TP = confm[0,0], confm[0,1], confm[1,0], confm[1,1]
    hh = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    MCC = (TP * TN - FP * FN) / np.sqrt(hh)
    print('MCC: ', MCC)

####RandomForest regression####
def random_forest_regressor(train_x, train_y):
    from sklearn import ensemble
    model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=2000)
    # 2000 decision trees are used here
    model_RandomForestRegressor.fit(train_x, train_y)
    return model_RandomForestRegressor

####RandomForest classification####
def random_forest_classifier(train_x, train_y):
    from sklearn import ensemble
    model_RandomForestClassifier = ensemble.RandomForestClassifier(n_estimators=2000)
    model_RandomForestClassifier.fit(train_x, train_y)
    return model_RandomForestClassifier

def read_data(group = 'SPL'):
    X = np.load('../data/X_loc_0.25day.npy')
    y = np.load('../data/yNew_loc_0.25day.npy')
    forSOD = False
    L = X.shape[0]
    Loc = [93]
    Sgroup = [i for i in range(13)]
    Pgroup = [i for i in range(13,93)]
    groups = {
        'S' : Sgroup,
        'P' : Pgroup,
        'SL' : Sgroup + Loc,
        'PL' : Pgroup + Loc,
        'SP' : Sgroup + Pgroup,
        'SPL' : Sgroup + Pgroup + Loc,
    }
    assert group in groups
    useGroup = groups[group]
    if forSOD == True:
        useGroup = [i for i in range(49)]
    X = X[:,:,useGroup]
    # **********************
    X = X.reshape(L, -1)
    y = y.reshape(L,)

    seed = 15
    np.random.seed(seed)
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    print('KFold = %d :'%n_splits)
    for train_index, test_index in kfold.split(X, y):
        train_X, test_X = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]
        model_fit(train_X, train_y, test_X, test_y)

    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # return train_X, train_y, test_X, test_y


def model_fit(train_X, train_y, test_X, test_y):
    model_save_file = ''
    model_save = {}

    test_regressor = ['RF']
    regressors = {
                   'RF': random_forest_classifier,
                   }
    print('reading training and testing data...')
    # train_x, train_y, test_x, test_y = read_data()

    for regressor in test_regressor:
        print('******************* %s ********************' % regressor)
        start_time = time.time()
        model = regressors[regressor](train_X, train_y)
        print('training took %fs!' % (time.time() - start_time))
        predict = model.predict(test_X)
        # print('test_y: {}\npredict: {}'.format(test_y, predict))
        # score = model.score(test_x, test_y)
        calMetrix(predict, test_y)
        plt.figure()
        plt.plot(np.arange(len(predict)), test_y, 'go-', label='true value')
        plt.plot(np.arange(len(predict)), predict, 'ro-', label='predict value')
        plt.title('%s' % regressor)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    read_data()