from keras.layers import Dense, LSTM
# from keras.layers import Input, Dropout, Activation
# from keras.layers import Convolution2D, MaxPooling2D, Reshape
from keras.layers import Convolution1D, MaxPooling1D, Flatten
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
import numpy as np
# from matplotlib import pyplot as plt
# import random
from sklearn import metrics
import argparse

pt = lambda s:print(type(s),s)

nb_epoch = 100
# number_of_batch = 100
batch_size = 72
# frame_row = 28
# frame_col = 2
# channels = 1
output_dim = 1

def classifyRes(arr):
    arr = arr.reshape(arr.size,)
    for i in range(arr.size):
        arr[i] = 1 if arr[i]>0.5 else 0
    return arr

def calMetrix(y_pre, test_y):
    print('******************* CNN-LSTM********************')
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

def read_data(xpath, ypath, group):
    # X = np.load('../data/X_loc.npy')
    # y = np.load('../data/yNew_loc.npy')
    X = np.load(xpath)
    y = np.load(ypath)
    L = X.shape[0]
    # **********************
    # smart: 0 - 12
    # perf: 13 - 92
    # loc: 93
    Loc = [93]
    Sgroup = [i for i in range(12)]
    Pgroup = [i for i in range(13, 93)]
    groups = {
        'S': Sgroup,
        'P': Pgroup,
        'SL': Sgroup + Loc,
        'PL': Pgroup + Loc,
        'SP': Sgroup + Pgroup,
        'SPL': Sgroup + Pgroup + Loc,
    }
    assert group in groups
    useGroup = groups[group]
    X = X[:, :, useGroup]
    m,n,p = X.shape
    X = X.reshape(m,n,p,1)
    # **********************
    # X = X.reshape(L, -1)
    y = y.reshape(L, -1)
    # ****************************

    from sklearn.model_selection import KFold
    seed = 15
    np.random.seed(seed)
    kfold = KFold(n_splits = 5, shuffle=True, random_state=seed)

    for train_index, test_index in kfold.split(X, y):
        train_X, test_X = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]
        print(train_X.shape)
        model_fit(train_X, train_y, test_X, test_y)
        # break

    # ****************************
    # train_X = X[:int(L * 0.9)]
    # train_y = y[:int(L * 0.9)]
    # test_X = X[int(L * 0.9):]
    # test_y = y[int(L * 0.9):]
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # return

def model_fit(train_X, train_y, test_X, test_y):
    # define model
    model = Sequential()
    # model.add(TimeDistributed(cnn))
    model.add(TimeDistributed(Convolution1D(128, 4, border_mode='same'), input_shape=train_X.shape[1:]))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(128, return_sequences=True, name="lstm_layer0"))
    model.add(LSTM(128, return_sequences=False, name="lstm_layer1"))
    # model.add(LSTM(100, return_sequences=True, name="lstm_layer2"))
    model.add(Dense(output_dim, activation='sigmoid'))
    # model.add(GlobalAveragePooling1D(name="global_avg"))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    # %%
    model.fit(train_X, train_y,batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, validation_data=(test_X, test_y))

    test_y = test_y.reshape(test_y.size, 1)
    predict_y = model.predict(test_X)
    predict_y = predict_y.reshape(predict_y.size, 1)
    predict_y = classifyRes(predict_y)
    calMetrix(predict_y, test_y)
    model.save('../model/CNN_LSTM_model.h5')

    # plt.plot(y_predict, 'r',label='forecast')
    # plt.plot(y_test, 'b',label='actual')
    # plt.legend()
    # plt.grid()
    # plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="train data path")
    parser.add_argument("--xpath", help="Xdata Path", default='', type=str,  required=True)
    parser.add_argument("--ypath", help="ydata Path", default='', type=str, required=True)
    parser.add_argument("--group", help="group", default='', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    read_data(args.xpath, args.ypath, args.group)
    # python3 CNNLSTM_args.py --xpath X1_10days.npy --ypath y1_10days.npy --group SPL
