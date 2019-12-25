# from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,TimeDistributed
from keras.layers import LSTM
import numpy as np
from sklearn import metrics
import argparse

def classifyRes(arr):
    arr = arr.reshape(arr.size,)
    for i in range(arr.size):
        arr[i] = 1 if arr[i]>0.5 else 0
    return arr

def calMetrix(y_pre, test_y):
    print('******************* LSTM********************')
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

def read_data(xpath, ypath, group = 'SPL'):
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
    # **********************
    # X = X.reshape(L, -1)
    y = y.reshape(L, -1)

    from sklearn.model_selection import KFold
    seed = 15
    np.random.seed(seed)
    kfold = KFold(n_splits = 5, shuffle=True, random_state=seed)

    for train_index, test_index in kfold.split(X, y):
        train_X, test_X = X[train_index], X[test_index]
        train_y, test_y = y[train_index], y[test_index]
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
    # train_X, train_y, test_X, test_y = read_data()
    outdim = train_y.shape[1]
    validation_number = int(0.9 * train_X.shape[0])
    validation_X = train_X[validation_number:]
    validation_y = train_y[validation_number:]
    train_X = train_X[:validation_number]
    train_y = train_y[:validation_number]

    model = Sequential()
    model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True, dropout=0.25))
    # print(model.layers)
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    # model.add(Dense(outdim))
    model.add(Dense(outdim, activation='sigmoid'))
    # model.add(Dense(outdim, activation='softmax'))
    # sigmoid and softmax are activation functions used by the neural network output layer
    # for binary discrimination and multi-class discrimination
    # binary cross-entropy and categorical cross-entropy are corresponding loss functions

    # model.add(Activation('linear'))

    # model.compile(loss='mse', optimizer='rmsprop')
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  # loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )

    # fit network
    history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(validation_X, validation_y),
                        verbose=0,
                        shuffle=True)

    # summarize performance of the model
    # scores = model.evaluate(train_X, train_y, verbose=0)
    # print(model.metrics_names)
    # print("model loss: %.2f%%" % (scores*100))

    # plot history
    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='validation')
    # plt.legend()
    # plt.grid()
    # plt.savefig('loss.png')
    # plt.clf()
    np.savetxt('loss.csv', history.history['loss'])
    np.savetxt('val_loss.csv', history.history['val_loss'])

    model.save('../model/keras_LSTM_model.h5')

    # make a prediction
    y_pre = model.predict(test_X)
    # yhat = yhat.reshape(train_y.shape[1])
    print(y_pre.shape)
    y_pre = classifyRes(y_pre)
    calMetrix(y_pre, test_y)

    # inv_yhat = yhat
    # inv_test_y = test_y
    np.savetxt('forecast.csv', y_pre)
    np.savetxt('actual.csv', test_y)

    # for k in range(inv_yhat.shape[1]):
    #     plt.plot(inv_yhat[:, k], label='forecast')
    #     plt.plot(inv_test_y[:, k], label='actual')
    #     plt.legend()
    #     plt.grid()
    #     plt.savefig( 'var_'+str(k)+'.png')
    #     plt.clf()
    #
    # plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="train data path")
    parser.add_argument("--xpath", help="Xdata Path", default='', type=str,  required=True)
    parser.add_argument("--ypath", help="ydata Path", default='', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    read_data(args.xpath, args.ypath)

