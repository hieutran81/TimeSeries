from matplotlib import pyplot as plt
import pandas
from pandas import DataFrame
from pandas import concat
from pandas import Series
import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.svm import SVR



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

def fit_svr(train):
    X, y = train[:, 0:-1], train[:,-1]
    model = SVR()
    model.fit(X,y)
	# prob = svm_problem(y,x)

    return model

def forecast_svr(model, X):
    X = X.reshape(1, len(X))
    yhat = model.predict(X)
    return yhat[0]


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

#fit the model
svr_model = fit_svr(train_scaled)
#forecast model entire training set to build up state

	#walk forward validation on test data
predictions = []
for i in range(len(test_scaled)):
	#make one step forecast
	X, y = test_scaled[i,0:-1], test_scaled[i,-1]
	yhat = forecast_svr(svr_model,X)
	#invert scaling
	yhat = invert_scale(scaler, X, yhat)
	#invert difference
	yhat = inverse_difference(raw_values,yhat,len(test_scaled)+1-i)
	#store forecast
	predictions.append(yhat)
	expected = raw_values[len(train_scaled)+i+1][0]
	print("Month = %d, predict: %f, expect: %f" %((i+1), yhat, expected))

rmse = sqrt(mean_squared_error(raw_values[-12:],predictions))
print(" Test RMSe : %f" %(rmse))

plt.plot(raw_values[-12:])
plt.plot(predictions)
plt.show()








