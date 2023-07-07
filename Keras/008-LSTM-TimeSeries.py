import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pylab

# Converte uma matrix de valores em dados de series temporais
def cria_timeseries(series, ts_lag=1):
    data_x = []
    data_y = []
    n_rows = len(series)-ts_lag
    for i in range(n_rows-1):
        a = series[i:(i+ts_lag), 0]
        data_x.append(a)
        data_y.append(series[i + ts_lag, 0])
    x, y = np.array(data_x), np.array(data_y)
    return x, y

# carregando o dataset
dataframe = read_csv('sp500.csv', usecols=[0])
plt.plot(dataframe)
plt.show()

# mudando o tipo
series = dataframe.values.astype('float32')

# Normalizando
scaler = StandardScaler()
series = scaler.fit_transform(series)

# dividindo em treino e teste
train_size = int(len(series) * 0.75)
test_size = len(series) - train_size
train, test = series[0:train_size, :], series[train_size:len(series), :]

# reshape
ts_lag = 1
trainX, trainY = cria_timeseries(train, ts_lag)
testX, testY = cria_timeseries(test, ts_lag)

# reshape para entrada dos dados
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Definindo o modelo
model = Sequential()
model.add(LSTM(10, input_shape=(1, ts_lag)))
model.add(Dense(1))
model.compile(loss='mean_squared_logarithmic_error', optimizer='adagrad')
model.fit(trainX, trainY, epochs=500, batch_size=30)

# Fazendo previsao
trainPredic = model.predict(trainX)
testPredic = model.predict(testX)

# Voltando para escala normal
trainPredic = scaler.inverse_transform(trainPredic)
trainY = scaler.inverse_transform(trainY)
testPredic = scaler.inverse_transform(testPredic)
testY = scaler.inverse_transform([testY])

# calculando o erro
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredic[:0]))
print('Score em treino: %.2f RMSE' % (trainScore))

pylab.plot(trainPredic)
pylab.plot(testPredic)
pylab.show()

