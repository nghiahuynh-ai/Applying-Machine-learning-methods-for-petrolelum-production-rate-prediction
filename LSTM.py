import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from math import sqrt
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


def invert_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def build_dataset(data, steps=1):
    data = np.insert(data, [0] * steps, 0)
    X, Y = [], []
    for i in range(len(data) - steps):
        X.append(data[i:(i + steps)])
        Y.append(data[i + steps])
    Y = np.array(Y)
    Y = np.reshape(Y, (Y.shape[0], 1))
    dataset = np.concatenate((X, Y), axis=1)
    return dataset


def split_dataset(dataset, train_size):
    split_point = int(dataset.shape[0] * train_size)
    trainData, testData = dataset[0:split_point], dataset[split_point:]
    return trainData, testData


def scale(trainData, validData, testData):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(trainData)
    trainData = trainData.reshape(trainData.shape[0], trainData.shape[1])
    trainDF = scaler.transform(trainData)
    validData = validData.reshape(validData.shape[0], validData.shape[1])
    validDF = scaler.transform(validData)
    testData = testData.reshape(testData.shape[0], testData.shape[1])
    testDF = scaler.transform(testData)
    return scaler, trainDF, validDF, testDF


def invert_scale(scaler, Xtrain, Ypredict):
    new_row = [x for x in Xtrain] + [Ypredict]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    Yinvert = scaler.inverse_transform(array)
    return Yinvert[0, -1]


def training(Xtrain, Ytrain, batch_size, epochs, neurals):
    Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], 1)
    model = Sequential()
    model.add(LSTM(units=neurals, activation='sigmoid',
                   batch_input_shape=(batch_size, Xtrain.shape[1], Xtrain.shape[2]), stateful=True))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(epochs):
        print('Epoch:', i + 1)
        model.fit(Xtrain, Ytrain, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        model.reset_states()
    return model


def predict(model, batch_size, Xtest):
    Xtest = Xtest.reshape(1, len(Xtest), 1)
    yhat = model.predict(Xtest, batch_size=batch_size)
    return yhat[0, 0]

def errorMeasure(actual, prediction):
    score = r2_score(actual, prediction)
    rmse = sqrt(mean_squared_error(actual, prediction))
    mae = mean_absolute_error(actual, prediction)
    return score, rmse, mae


def graph(time, series, XtrainLen, trainPredict, XvalidLen, validPredict, testPredict):
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.plot(time, series.values, label='Actual Value', color='purple')
    ax.plot(time.values[0:XtrainLen], trainPredict, label='Train Value', color='blue')
    ax.plot(time.values[XtrainLen:XtrainLen + XvalidLen], validPredict, label='Validation Value', color='orange')
    ax.plot(time.values[XtrainLen + XvalidLen:len(series) - 1], testPredict, label='Test Value', color='green')
    ax.set_title('LSTM MODEL')
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.set_xlabel('Time (day)', fontsize=16)
    ax.set_ylabel('Oil Rate' + r'$(m^3)$', fontsize=16)

    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.scatter(series.values[0:XtrainLen], trainPredict, label='Train Value', color='blue')
    ax.scatter(series.values[XtrainLen:XtrainLen + XvalidLen], validPredict, label='Validation Value', color='orange')
    ax.scatter(series.values[XtrainLen + XvalidLen:len(series) - 1], testPredict, label='Test Value', color='green')
    ax.set_title('ACTUAL VS PREDICTED VALUES')
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_xlabel('Actual Value', fontsize=16)
    ax.set_ylabel('Predicted Value', fontsize=16)

    actualCumlative = list()
    for i in range(len(series.values)):
        actualCumlative.append(sum(series['OIL'].values[0:i]))

    prediction = pd.DataFrame(np.concatenate((trainPredict, validPredict, testPredict), axis=0), columns=['OIL'])
    predictedCumulative = list()
    for i in range(len(prediction)):
        predictedCumulative.append(sum(prediction['OIL'].values[0:i]))

    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.plot(time, actualCumlative, label='Train Value', color='blue')
    ax.plot(time['STEP'].values[1:len(time)], predictedCumulative, label='Validation Value', color='orange')
    ax.set_title('CUMULATIVE PRODUCTION OIL RATE')
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_xlabel('Time (days)', fontsize=16)
    ax.set_ylabel('Oil Rate' + 'r$(m^3)$', fontsize=16)

    plt.show()


def assemble(series, steps, train_size, batch_size, neurals, epochs, time):
    diff = difference(series.values)
    dataset = build_dataset(data=diff.values, steps=steps)
    trainData, validTest = split_dataset(dataset, train_size)
    validData, testData = split_dataset(validTest, 0.5)

    scaler, Strain, Svalid, Stest = scale(trainData, validData, testData)
    Xtrain, Ytrain = Strain[:, 0:-1], Strain[:, -1]
    Xvalid, Yvalid = Svalid[:, 0:-1], Svalid[:, -1]
    Xtest, Ytest = Stest[:, 0:-1], Stest[:, -1]

    lstm = training(Xtrain, Ytrain, batch_size, epochs, neurals)

    train_prediction = list()
    for i in range(len(Xtrain)):
        Yhat = predict(lstm, 1, Xtrain[i])
        Yhat = invert_scale(scaler, Xtrain[i], Yhat)
        Yhat = invert_difference(series.values, Yhat, len(series.values) - i)
        train_prediction.append(Yhat)
    trainScore, trainRmse, trainMae = errorMeasure(series.values[0:len(Xtrain)], train_prediction)

    valid_prediction = list()
    for i in range(len(Xvalid)):
        Yhat = predict(lstm, 1, Xvalid[i])
        Yhat = invert_scale(scaler, Xvalid[i], Yhat)
        Yhat = invert_difference(series.values, Yhat, len(Xvalid) + len(Xtest) + 1 - i)
        valid_prediction.append(Yhat)
    validScore, validRmse, validMae = errorMeasure(series.values[len(Xtrain):len(Xtrain) + len(Xvalid)],
                                                   valid_prediction)

    test_prediction = list()
    for i in range(len(Xtest)):
        Yhat = predict(lstm, 1, Xtest[i])
        Yhat = invert_scale(scaler, Xtest[i], Yhat)
        Yhat = invert_difference(series.values, Yhat, len(Xtest) + 1 - i)
        test_prediction.append(Yhat)
    testScore, testRmse, testMae = errorMeasure(series.values[len(Xtrain) + len(Xvalid):len(series) - 1],
                                                test_prediction)

    actualTotalFlow = sum(series.values)
    predictedTotalFlow = sum(train_prediction) + sum(valid_prediction) + sum(test_prediction)

    print('Train Score:         ', trainScore)
    print('Train RMSE:          ', trainRmse)
    print('Train MAE:           ', trainMae)
    print('Validation Score:    ', validScore)
    print('Validation RMSE:     ', validRmse)
    print('Validation MAE:      ', validMae)
    print('Test Score:          ', testScore)
    print('Test RMSE:           ', testRmse)
    print('Test MAE:            ', testMae)
    print('Actual Total Flow    ', actualTotalFlow)
    print('Predicted Total Flow ', predictedTotalFlow)

    graph(time, series, len(Xtrain), train_prediction, len(Xvalid), valid_prediction, test_prediction)

    dt1, dt2, dt3 = train_prediction, valid_prediction, test_prediction
    prediction = pd.DataFrame(np.concatenate((dt1, dt2, dt3), axis=0))

    predictedCumulative = list()
    for i in range(len(prediction)):
        predictedCumulative.append(sum(prediction.values[0:i]))

    sqrError = list()
    for i in range(len(prediction)):
        sqrError.append((prediction.values[i] - series.values[i])**2)

    absError = list()
    for i in range(len(prediction)):
        absError.append(abs(prediction.values[i] - series.values[i]))

    output = pd.DataFrame(np.concatenate((time.values[0:len(time) - 1], prediction), axis=1), columns=['STEP', 'OIL'])
    output['CUMULATIVE'] = pd.DataFrame(predictedCumulative)
    output['SQUARED ERROR'] = pd.DataFrame(sqrError)
    output['ABSOLUTE ERROR'] = pd.DataFrame(absError)

    output.to_csv('Volve_well/outputGRU.csv')
    output.to_excel('Volve_well/outputGRU.xlsx')

    lstmModel = lstm.to_json()
    with open('lstmModel.json', 'w') as json_file:
        json_file.write(lstmModel)
    lstm.save_weights("lstmWeight.h5")


def run():
    rawData = pd.read_csv('F14.csv')
    series = rawData[['OIL']]
    time = rawData[['STEP']]
    assemble(series=series, steps=5, train_size=0.7, batch_size=1, neurals=1, epochs=10, time=time)


run()
