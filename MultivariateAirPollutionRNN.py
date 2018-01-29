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
import tensorflow as tf


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
    print(dataset.head(5))
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
    # integer encode direction column 4 with is wind
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    # ensure all is float
    values = values.astype("float32")

    # normailize feature
    scaler = MinMaxScaler(feature_range=(0, 1))
    values_scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframe = series_to_supervised(values_scaled, n_hours, 1)
    # drop column don't need to predict
    reframe.drop(reframe.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
    print(reframe.head())
    return reframe.values,scaler

def split(values):
    train = values[:n_train, :]
    test = values[n_train:, :]

    # split into input and output
    train_X, train_y = train[:, :n_obs], train[:, -n_features]
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
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    return model, history

def fit_rnn():
    global X, y, cell, outputs, states, real, loss, optimizer, training_op, init
	# data
    print(train_X.shape)
    print(train_y.shape)
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None])
    # define network
    cell = tf.contrib.rnn.OutputProjectionWrapper(tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu), output_size=n_outputs)
    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    real = tf.contrib.layers.fully_connected(states, n_outputs, activation_fn = None)

    loss = tf.reduce_mean(tf.square(real - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for i in range(0, n_train, batch_size):
                end = i + batch_size
                X_batch, y_batch = train_X[i:end, :, :], train_y[i:end]
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            mse = loss.eval(feed_dict={X: train_X, y: train_y})
            #print(epoch, "\tMSE:", mse)
        saver.save(sess,"./multivariate_model.ckpt")

def forecast_rnn(X_test):
    global outputs,init, X
    print("x test")
    print(X_test.shape)
    saver = tf.train.Saver()
    X_test = X_test.reshape(X_test.shape[0],n_steps , n_inputs)
    print(X_test.shape)
    with tf.Session() as sess:
        saver.restore(sess,save_path="./multivariate_model.ckpt")
        sess.run(init)
        print(type(X))
        sess.run(real, feed_dict = {X: X_test})
        y_pred = real.eval(feed_dict = {X: X_test})
    return y_pred



def invert_scale():
    # invert scaling for forecast
    print(yhat.shape)
    print(test_X.shape)
    inv_yhat = concatenate((yhat, test_X[:, -7:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    inv_y = concatenate((test_y, test_X[:, -7:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    return inv_y,inv_yhat

#load dataset
dataset = read_csv('pollution.csv',header = 0, index_col=0)
# values is numpy array shape (43800,8)
values = dataset.values

# specify number of lag hour
n_hours = 3
n_features = 8
n_obs = n_hours * n_features

#preprocess to ready to train
values,scaler = preprocess(values)



# split into train and test set
n_train = 365 * 24
train_X, train_y, test_X, test_y = split(values)

# config networks
n_train = len(train_X)
n_steps = 3
n_inputs = 8
n_neurons = 100
n_outputs = 1
learning_rate = 0.001
batch_size = 72
n_epochs = 50



fit_rnn()

# make a prediction
yhat = forecast_rnn(test_X)
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
print('Test RMSE: %.3f' % rmse)
plt.plot(inv_y[:100])
# plt.title("actual")
# plt.figure(2)
plt.plot(inv_yhat[:100])
plt.title("forecast")
plt.show()
