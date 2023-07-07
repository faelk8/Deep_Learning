import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
# para as dimensões
from keras import backend as k
#k.set_image_dim_ordering('th')

# Carregando o dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Processamento da entrada
# reshape sample, channels, width, heigth
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

# Convertando para float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print(x_train.shape)

#  Normalização - Z-score ou Gaussiana
x_train = x_train - np.mean(x_train) / x_train.std()
x_test = x_test - np.mean(x_test) / x_test.std()
# 60000, 1, 28, 28

# Convertendo 1-dim para 10-dim / one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
print(num_classes)

# Definindo a amostra
print(x_train.shape)

# Definindo o modelo
model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(240, activation='elu'))
model.add(Dense(num_classes, activation='softmax'))
print(model.output_shape)
model.compile(loss='binary_crossentropy', optimizer='adagrad', matrices=['acc'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=200)

# Fazendo previsões
scores = model.evaluate(x_test, y_test, verbose=0)
print("CNN error: % .2f%%" % (100-scores[1]*100))