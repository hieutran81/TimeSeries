from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot as plt
import pandas
from pandas import DataFrame
from pandas import concat
from pandas import Series
import numpy
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
# from keras.layers import LSTM
# from keras.layers import Dense
# from keras.models import Sequential


# load dataset
def loadAndVisualizeData():
    dataframe = pandas.read_csv('sales-of-shampoo-over-a-three-ye.csv', usecols=[1], engine='python', skipfooter=3)
    dataset = dataframe.values
    # plt.plot(dataframe)
    # plt.show()
    return dataset

def walk_forward():
    # walk-forward validation
    history = [x for x in train]
    predictions = list()
    for i in range(len(test)):
        # make prediction
        predictions.append(history[-1])
        # observation
        history.append(test[i])
        # report performance

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = DataFrame(data)
	columns = [df.shift(i) for i in range(1, lag+1)]
	columns.append(df)
	df = concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df

# create  a diffrerenced between value
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value[0])
	return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	train_scaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	test_scaled = scaler.transform(test)
	return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
	new_row = [x for x in X] + [value]
	array = numpy.array(new_row)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# def fit_lstm(train, batch_size, nb_epoch, neurons):
# 	X, y = train[:, 0:-1], train[:,-1]
# 	X = X.reshape(X.shape[0],1,X.shape[1])
# 	model = Sequential()
# 	model.add(LSTM(neurons,batch_input_shape=(batch_size,X.shape[1],X.shape[2]), stateful=True))
# 	model.add(Dense(1))
# 	model.compile(loss='mean_squared_error',optimizer='adam')
# 	for i in range(nb_epoch):
# 		model.fit(X,y,epochs=1, batch_size= batch_size, verbose= 0, shuffle=False )
# 		model.reset_states()
# 		print("epoch %d" %(i))
# 	return model

def fit_rnn(train, batch_size, nb_epoch, neurons):
    global X, y, cell, outputs, states, real, loss, optimizer, training_op, init
    # config networks
    n_train = len(train)
    n_steps = 1
    n_inputs = 1
    n_neurons = 100
    n_outputs = 1
    learning_rate = 0.001
	# data
    X_train, y_train = train[:, 0:-1], train[:, -1]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
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
        for epoch in range(nb_epoch):
            for i in range(0, n_train, batch_size):
                end = i + batch_size
                X_batch, y_batch = X_train[i:end, :, :], y_train[i:end]
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            mse = loss.eval(feed_dict={X: X_train, y: y_train})
            #print(epoch, "\tMSE:", mse)
        saver.save(sess,"./univariate_model.ckpt")

def forecast_rnn(X_test):
    global outputs,init, X
    saver = tf.train.Saver()
    X_test = X_test.reshape(X_test.shape[0], 1, 1)
    print(X_test.shape)
    with tf.Session() as sess:
        saver.restore(sess,save_path="./univariate_model.ckpt")
        sess.run(init)
        print(type(X))
        sess.run(real, feed_dict = {X: X_test})
        y_pred = real.eval(feed_dict = {X: X_test})
    return y_pred

def forecast_lstm(model, batch_size, X):
	X = X.reshape(1,1,len(X))
	yhat = model.predict(X, batch_size = batch_size)
	return yhat[0][0]

# load dataset
raw_values = loadAndVisualizeData()

# transform data to be stationary
diff_values = difference(raw_values,1)

# transform data to supervised learning
supervised = timeseries_to_supervised(diff_values,1)
supervised_values = supervised.values

# split into train and test set
train, test = supervised_values[:-12], supervised_values[-12:]

# transform scale data to [-1,1]
scaler, train_scaled, test_scaled = scale(train,test)

#repeat experiment
repeats = 1
error_scores = []
for r in range(repeats):
    fit_rnn(train_scaled,1,1500,4)

    # #fit the model
    # lstm_model = fit_lstm(train_scaled, 1, 150, 4)
    # #forecast model entire training set to build up state
    # train_reshape = train_scaled[:,0].reshape(len(train_scaled),1,1)
    # lstm_model.predict(train_reshape, batch_size=1)
    #walk forward validation on test data
    predictions = []
    for i in range(len(test_scaled)):
		#make one step forecast
        X_val, y_val = test_scaled[i,0:-1], test_scaled[i,-1]
        yhat = forecast_rnn(X_val)
        #invert scaling
        yhat = invert_scale(scaler, X_val, yhat)
        #invert difference
        yhat = inverse_difference(raw_values,yhat,len(test_scaled)+1-i)
        #store forecast
        predictions.append(yhat)
        expected = raw_values[len(train_scaled)+i+1][0]
        print("Month = %d, predict: %f, expect: %f" %((i+1), yhat, expected))

	# report performance
    rmse = sqrt(mean_squared_error(raw_values[-12:],predictions))
    print("%d Test RMSe : %f" %((r+1),rmse))
    error_scores.append(rmse)
    plt.plot(raw_values[-12:])
    plt.plot(predictions)
    plt.show()

#line plot to visualize
# summarize results
# results = DataFrame()
# results['rmse'] = error_scores
# print(results.describe())
# results.boxplot()


# plt.plot(raw_values[-12:])
# plt.plot(predictions)
# plt.show()









