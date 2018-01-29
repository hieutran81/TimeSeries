# load and plot dataset
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot as plt
from pandas import DataFrame
from pandas import concat
from pandas import Series
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from numpy import array
from math import sqrt
from sklearn.metrics import mean_squared_error

# load dataset
def loadAndVisualizeData():
    dataframe = read_csv('sales-of-shampoo-over-a-three-ye.csv', usecols=[1], engine='python', skipfooter=3)
    # plt.plot(dataframe)
    # plt.show()
    return dataframe

# convert time series into supervised learning problem
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

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i][0] - dataset[i - interval][0]
		diff.append(value)
	return Series(diff)

# transform data into train and test set
def prepare_data(series, n_test, n_lag, n_seq):
    # extract values
    raw_values = series.values
    # transform to stationary
    diff_series = difference(raw_values,1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values),1)

    # transform to scale [-1,1]
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaled_values = scaler.fit_transform(diff_values)

    #transform to  supervised learning
    supervised = series_to_supervised(scaled_values,n_lag,n_seq)
    supervised_values = supervised.values
    #print(supervised.head())

    # split into train and test set
    train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, train, test


# fit an LSTM network to training data
def fit_svr(train, n_lag, n_seq):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], X.shape[1])
    models = []
    for i in range(n_seq):
        yi = y[:,i]
        print(yi.shape)
        print(yi)
        model = SVR()
        model.fit(X,yi)
        models.append(model)
    return models

# make one forecast with an LSTM,
def forecast_svr(models, X):
	# reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, len(X))
    # make forecast
    forecast = []
    for model in models:
        forecast.append(model.predict(X))
    # convert to array
    return [x for x in forecast]

# evaluate the persistence model
def make_forecasts(models, test, n_lag, n_seq):
    forecasts = []
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = forecast_svr(models, X)
        print(type(forecast))
        print(forecast)
        # store the forecast
        forecasts.append(forecast)
    return forecasts

# invert differenced forecast
def inverse_difference(last_ob, forecast):
	# invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted

# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = []
    for i in range(len(forecasts)):
		# create array from forecast
        forecast = array(forecasts[i])
        forecast = forecast.reshape(1, len(forecast))
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index][0]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    # print(inverted)
    # print(type(inverted))
    return inverted

# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))

    # plot the forecasts in the context of the original dataset


def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    plt.plot(series.values)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        plt.plot(xaxis, yaxis, color='red')
    # show the plot
    plt.show()

# load data
series = loadAndVisualizeData()

#configure
n_lag = 1
n_seq = 3
n_test = 10
n_epochs = 1500
n_batchs = 1
n_neurons = 1

#prepare data
scaler, train, test = prepare_data(series, n_test,n_lag,n_seq)
# fit model
models = fit_svr(train, n_lag, n_seq)

# # make forecasts
forecasts = make_forecasts(models, test, n_lag, n_seq)
# inverse transform forecasts and test
forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
actual = [row[n_lag:] for row in test]
actual = inverse_transform(series, actual, scaler, n_test+2)
#evaluate forecast
evaluate_forecasts(actual,forecasts,n_lag,n_seq)
#plot forecast
plot_forecasts(series,forecasts,n_test+2)

