import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle


def loadData(fileName, labelName, featureName):
    dataFrame = pd.read_csv(fileName)
    feature = dataFrame[featureName]
    label = dataFrame[[labelName]]
    return feature, label


def processData(fraction, feature, label):
    X, xtest, Y, ytest = train_test_split(feature, label, test_size=fraction, shuffle=False)
    xtrain, xvalid, ytrain, yvalid = train_test_split(X, Y, test_size=fraction / (1 - fraction), shuffle=True)
    return xtrain, ytrain, xvalid, yvalid, xtest, ytest


def linearize(exp, ytrain, yvalid, ytest):
    ytrain = ytrain ** (1 / exp)
    yvalid = yvalid ** (1 / exp)
    ytest = ytest ** (1 / exp)
    return ytrain, yvalid, ytest


def train(xtrain, ytrain):
    learner = SVR()
    learner.fit(xtrain, ytrain.values.ravel())
    return learner


def predict(learner, xtrain, xvalid, xtest):
    train_pred = learner.predict(xtrain)
    valid_pred = learner.predict(xvalid)
    test_pred = learner.predict(xtest)
    return train_pred, valid_pred, test_pred


def invert(exp, train_pred, valid_pred, test_pred):
    train_pred = train_pred ** exp
    valid_pred = valid_pred ** exp
    test_pred = test_pred ** exp
    return train_pred, valid_pred, test_pred


def getAccuracy(ytrainln, train_pred, yvalidln, valid_pred, ytestln, test_pred):
    trainScore = metrics.r2_score(ytrainln, train_pred)
    validScore = metrics.r2_score(yvalidln, valid_pred)
    testScore = metrics.r2_score(ytestln, test_pred)
    result = {'Train Score': trainScore, 'Valid Score': validScore, 'Test Score': testScore}
    return result


def getRMSE(ytrain, train_pred, yvalid, valid_pred, ytest, test_pred):
    trainRMSE = np.sqrt(metrics.mean_squared_error(ytrain, train_pred))
    validnRMSE = np.sqrt(metrics.mean_squared_error(yvalid, valid_pred))
    testRMSE = np.sqrt(metrics.mean_squared_error(ytest, test_pred))
    result = {'Train RMSE': trainRMSE, 'Valid RMSE': validnRMSE, 'Test RMSE': testRMSE}
    return result


def getMAE(ytrain, train_pred, yvalid, valid_pred, ytest, test_pred):
    trainMAE = np.sqrt(metrics.mean_absolute_error(ytrain, train_pred))
    validnMAE = np.sqrt(metrics.mean_absolute_error(yvalid, valid_pred))
    testMAE = np.sqrt(metrics.mean_absolute_error(ytest, test_pred))
    result = {'Train MAE': trainMAE, 'Valid MAE': validnMAE, 'Test MAE': testMAE}
    return result


def getTotalPred(train_pred, valid_pred, test_pred):
    return sum(train_pred) + sum(valid_pred) + sum(test_pred)


def getPredictedSet(labelName, xtrainStep, train_pred, xvalidStep, valid_pred, xtestStep, test_pred):
    trainDF = np.concatenate((xtrainStep, pd.DataFrame(train_pred)), axis=1)
    validDF = np.concatenate((xvalidStep, pd.DataFrame(valid_pred)), axis=1)
    testDF = np.concatenate((xtestStep, pd.DataFrame(test_pred)), axis=1)
    predicted = pd.DataFrame(np.concatenate((trainDF, validDF, testDF), axis=0), columns=['STEP', labelName])
    predicted = predicted.sort_values(by='STEP')
    return predicted


def getCumulative(label, predicted):
    actual = list()
    for i in range(len(label.values)):
        actual.append(sum(label.values[0:i]))
    pred = list()
    for i in range(len(predicted.values)):
        pred.append(sum(predicted.values[0:i]))
    return pd.DataFrame(actual), pd.DataFrame(pred)


def saveOutput(learner, labelName, actual, predicted, cumulative):
    sqrError = list()
    for i in range(len(predicted.values)):
        sqrError.append((predicted[labelName].values[i] - actual[labelName].values[i]) ** 2)

    absError = list()
    for i in range(len(predicted.values)):
        absError.append(abs(predicted[labelName].values[i] - actual[labelName].values[i]))

    output = predicted
    output.reset_index(inplace=True)
    output.drop(columns='index', inplace=True)
    output['CUMULATIVE'] = cumulative
    output['SQUARED ERROR'] = pd.DataFrame(sqrError)
    output['ABSOLUTE ERROR'] = pd.DataFrame(absError)

    output.to_csv(labelName + 'rateSVR.csv')
    output.to_excel(labelName + 'rateSVR.xlsx')

    filename = labelName + 'SVR.sav'
    pickle.dump(learner, open(filename, 'wb'))


def printEvaluation(accuracy, rmse, mae, realTotalFlow, predTotalFlow):
    print('Train Score: ', accuracy['Train Score'])
    print('Valid Score: ', accuracy['Valid Score'])
    print('Test Score:  ', accuracy['Test Score'])
    print('Train RMSE:  ', rmse['Train RMSE'])
    print('Valid RMSE:  ', rmse['Valid RMSE'])
    print('Test RMSE:   ', rmse['Test RMSE'])
    print('Train MAE:   ', mae['Train MAE'])
    print('Valid MAE:   ', mae['Valid MAE'])
    print('Test MAE:    ', mae['Test MAE'])
    print('Actual Total Flow   : ', realTotalFlow)
    print('Predicted Total Flow: ', predTotalFlow)


def plotActualVsPred(labelName, ytrain, train_pred, yvalid, valid_pred, ytest, test_pred):
    plt.rc('font', size=12)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(ytrain, train_pred, color='tab:blue', label='Train Value')
    ax.scatter(yvalid, valid_pred, color='tab:green', label='Validation Value')
    ax.scatter(ytest, test_pred, color='tab:orange', label='Test Value')
    ax.set_title('ACTUAL VS PREDICTED ' + labelName + ' RATE VALUES - SVR')
    ax.set_xlabel('ACTUAL')
    ax.set_ylabel('PREDICTED')
    ax.grid(True)
    ax.legend()


def plotRateVsTime(labelName, step, label, trainStep, train_pred, validStep, valid_pred, testStep, test_pred):
    plt.rc('font', size=12)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(step, label, color='tab:purple', label='Actual Value')
    ax.scatter(trainStep, train_pred, color='tab:blue', label='Train Value')
    ax.scatter(validStep, valid_pred, color='tab:green', label='Validation Value')
    ax.scatter(testStep, test_pred, color='tab:orange', label='Test Value')
    ax.set_title(labelName + ' RATE PREDICTION BASED ON SUPPORT VECTOR REGRESSION')
    ax.set_xlabel('TIME (DAY)')
    ax.set_ylabel(labelName + ' RATE (CUBIC METER)')
    ax.grid(True)
    ax.legend()


def plotCumulative(labelName, step, realCumulative, predCumulative):
    plt.rc('font', size=12)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(step, realCumulative, color='tab:blue', label='Actual Value')
    ax.plot(step, predCumulative, color='tab:orange', label='Predicted Value')
    ax.set_title('CUMULATIVE ' + labelName + ' RATE - SVR')
    ax.set_xlabel('TIME (DAY)')
    ax.set_ylabel(labelName + ' RATE (CUBIC METER)')
    ax.grid(True)
    ax.legend()


def main():
    fileName = 'F14.csv'
    labelName = 'GAS'
    featureName = ['STEP', 'BTHP', 'DP', 'CS', 'WHP']
    exp = 4
    fraction = 0.15

    # prepare data
    feature, label = loadData(fileName, labelName, featureName)
    xtrain, ytrain, xvalid, yvalid, xtest, ytest = processData(fraction, feature, label)
    ytrainln, yvalidln, ytestln = linearize(exp, ytrain, yvalid, ytest)

    # train & predict
    learner = train(xtrain, ytrainln)
    train_pred, valid_pred, test_pred = predict(learner, xtrain, xvalid, xtest)

    # evaluate
    accuracy = getAccuracy(ytrainln, train_pred, yvalidln, valid_pred, ytestln, test_pred)
    train_pred, valid_pred, test_pred = invert(exp, train_pred, valid_pred, test_pred)
    rmse = getRMSE(ytrain, train_pred, yvalid, valid_pred, ytest, test_pred)
    mae = getMAE(ytrain, train_pred, yvalid, valid_pred, ytest, test_pred)

    # post-processing
    realTotalFlow = sum(np.array(label))
    predTotalFlow = getTotalPred(train_pred, valid_pred, test_pred)

    prediction = getPredictedSet(labelName, xtrain[['STEP']], train_pred, xvalid[['STEP']],
                                 valid_pred, xtest[['STEP']], test_pred)
    realCumulative, predCumulative = getCumulative(label, prediction[labelName])

    saveOutput(learner, labelName, label, prediction, predCumulative)

    printEvaluation(accuracy, rmse, mae, realTotalFlow, predTotalFlow)

    plotActualVsPred(labelName, ytrain, train_pred, yvalid, valid_pred, ytest, test_pred)
    plotRateVsTime(labelName, feature['STEP'], label, xtrain['STEP'], train_pred,
                   xvalid['STEP'], valid_pred, xtest['STEP'], test_pred)
    plotCumulative(labelName, np.array(feature['STEP']), np.array(realCumulative), np.array(predCumulative))

    plt.show()


if __name__ == "__main__":
    main()
