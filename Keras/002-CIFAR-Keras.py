import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

# ==== Carregando os dados ====
np.random.seed(100)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# ==== Pre processamento ====
x_train = x_train.reshape(50000, 3072) # 50.000 imagens para treino
x_test = x_test.reshape(10000, 3072) # 10.000 imagens para o teste

# Normalizando os dados Gaussiana
x_train = (x_train - np.mean(x_train))/np.std(x_train)
x_test = (x_test - np.mean(x_test))/np.std(x_test)

# Convertendo para one hot
labels = 10
y_train = np_utils.to_categorical(y_train, labels)
y_test = np_utils.to_categorical(y_test, labels)

# ==== Parando o treinamento ====
stopping = EarlyStopping(patience=2)

# ==== Definindo o modelo ====
model = Sequential()
model.add(Dense(512, input_shape=(3072,))) # 3*32*32 = 3072
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(labels)) # quantidade de saída dos dados a serem previsto
model.add(Activation('sigmoid'))

# Compilando o modelo
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
# Fit
model.fit(x_train, y_train, batch_size=100, epochs=100, validation_data=(x_test, y_test)) # , callbacks=[stopping])

# ==== Avalianda o modelo ====
score = model.evaluate(x_test, y_test, verbose=0)
print("Acurácia em teste: ", score[1])

# ==== Fazendo previsão ====
model.predict_classes(x_test)

# ==== Salvando e carregando o modelo ====
model.save('model.h5')
jsonmodel = model.to_json()
model.save_weights('modelWeight.h5')

#modelWt = model.weights('modelweight.h5')