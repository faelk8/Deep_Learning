from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
import numpy as np

# definindo lote, passo e classe para ser prevista
batch_size = 64
num_classes = 10
epochs = 20

# carrega os dados e mudando o formato
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784) # 28*28
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizando os dados guassiana
x_train = (x_train - np.mean(x_train))/np.std(x_train)
x_test = (x_test - np.mean(x_test))/np.std(x_test)

print(x_train.shape[0], ' amostra de treino')
print(x_test.shape[0], ' amostra de teste')

# converte para o tipo categorico
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

# definindo o modelo
model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10)) # uma saída por classe
model.add(Activation('softmax'))
# optimazor
rms = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['acc'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# Fazendo a previsão
score = model.evaluate(x_test, y_test, verbose=0)
print('Teste: ', score[0])
print('Treino ', score[1])
