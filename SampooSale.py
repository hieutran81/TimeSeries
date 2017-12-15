# load and plot dataset
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot as plt
import pandas
from math import sqrt
from sklearn.metrics import mean_squared_error


# load dataset
def loadAndVisualizeData():
    dataframe = pandas.read_csv('sales-of-shampoo-over-a-three-ye.csv', usecols=[1], engine='python', skipfooter=3)
    dataset = dataframe.values
    # print(dataset)
    # print(dataset.shape)
    # print(dataset[1].shape)
    # plt.plot(dataframe)
    # plt.show()
    return dataset

X = loadAndVisualizeData()
train, test = X[0:-12], X[-12:]
# walk-forward validation
history = [x for x in train]
print(history[0].shape)
predictions = list()
for i in range(len(test)):
	# make prediction
	predictions.append(history[-1])
	# observation
	history.append(test[i])
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
print(predictions)
# line plot of observed vs predicted
# plt.plot(test)
# plt.plot(predictions)
# plt.show()