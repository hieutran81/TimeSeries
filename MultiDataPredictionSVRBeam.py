from pandas import read_csv
from datetime import datetime
from pandas import DataFrame
from pandas import concat
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
# from keras.models import  Sequential
# from keras.layers import Dense
# from keras.layers import SimpleRNN
# from keras.layers import LSTM
from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from Dataset_Prediction import *
from sklearn.utils import check_array
from sklearn.svm import SVR


def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')

def preprocessData():
    # load data
    dataset = read_csv('PRSA_data_2010.1.1-2014.12.31.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)
    # manually specify column names
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    # mark all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)
    # drop the first 24 hours
    dataset = dataset[24:]
    # summarize first 5 rows
    # print(dataset.head(5))
    # save to file
    dataset.to_csv('pollution.csv')

def visuallize():
    # load dataset
    dataset = read_csv('pollution.csv', header=0, index_col=0)
    values = dataset.values
    # specify columns to plot
    groups = [0, 1, 2, 3, 5, 6, 7]
    i = 1
    # plot each column
    plt.figure()
    for group in groups:
        plt.subplot(len(groups), 1, i)
        plt.plot(values[:, group])
        plt.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    plt.show()

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def preprocess(values):
    # ensure all is float
    values = values.astype("float32")

    # normailize feature
    scaler = MinMaxScaler(feature_range=(0, 1))
    values_scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframe = series_to_supervised(values_scaled, n_hours, 1)
    # print(reframe.values.shape)
    # drop column don't need to predict
    #reframe.drop(reframe.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    return reframe.values,scaler

def split(values):
    train = values[:n_train, :]
    test = values[n_train:, :]

    # split into input and output
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
    # print(train_X.shape)
    # print(train_y.shape)
    # print(train_y)

    test_X, test_y = test[:, :n_obs], test[:, -n_features]

    # reshape to be 3d [sample, time step, feature]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    return train_X, train_y, test_X, test_y


def fit_lstm():
    # design network
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    #model.add(SimpleRNN(50, input_shape=(train_X.shape[1], train_X.shape[2])))

    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=100, batch_size= 72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    return model, history

def fit_svr(train_X):
    # design model
    # global train_X
    model = SVR(C = 1.1, epsilon=0.01)
    train_X = train_X.reshape((train_X.shape[0],n_obs))
    model.fit(train_X,train_y)
    return model

def invert_scale():
    # invert scaling for forecast
    # print(yhat.shape)
    # print(test_X.shape)
    shift = -(n_features-1)
    inv_yhat = concatenate((yhat, test_X[:, shift:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    inv_y = concatenate((test_y, test_X[:, shift:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    return inv_y,inv_yhat

def visualizeHistory(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()

def cal_mape(y_true, y_pred):
    # y_true, y_pred = check_array(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#load dataset
#dataset = read_csv('pollution.csv',header = 0, index_col=0)
# values is numpy array
values = data_in_hour()
n_train = 365 * 24
#values[:n_train] = replace_abnormal_detection_in_hour(values[:n_train])

# specify number of lag hour
n_hours = 3
n_features = 6
n_obs = n_hours * n_features

#preprocess to ready to train
values,scaler = preprocess(values)

# split into train and test set
#365 #* 24
train_X, train_y, test_X, test_y = split(values)

#model, history = fit_lstm()
model = fit_svr(train_X)
# visualizeHistory(history)
# make a prediction
# print(test_X.shape)
test_X = test_X.reshape((test_X.shape[0],n_obs))
yhat = model.predict(test_X)
yhat = yhat.reshape(yhat.shape[0],1)
# print("yhat")
# print(yhat)
# print(yhat.shape)

#invert to compute rmse
test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
test_y = test_y.reshape((len(test_y), 1))
inv_y, inv_yhat = invert_scale()
# print(inv_yhat)
# print(inv_y)

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
mape = cal_mape(inv_y, inv_yhat)
print('Test RMSE: %.3f' % rmse)
print('Test MAPE: %.3f' % mape)
plt.plot(inv_y[:50])
# plt.title("actual")
# plt.figure(2)
plt.plot(inv_yhat[:50])
plt.title("forecast")
plt.show()
