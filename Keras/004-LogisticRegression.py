from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils

iris = load_iris()

# sera usado 4 atributos para previsão
x, y = iris.data[:, :4], iris.target

# dividindo em dados de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, random_state=0)
'''# treinando o modelo
lr = LogisticRegressionCV
lr.fit(x_train, y_train)
pred_y = lr.predict(x_test)
print('Teste: {:.2f}'.format(lr.score(x_test, y_test)))'''

# transformando em one hot
def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

# dividindo en teste e treino
y_one = one_hot_encode_object_array(y_train)
y_one = one_hot_encode_object_array(y_test)

# criando o modelo
model = Sequential()
model.add(Dense(16, input_shape=(4,)))
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
model.fit(x_train, y_one, verbose=1, batch_size=1, epochs=1000)

score, accuracy = model.evaluate(x_test, y_one, batch_size=16, verbose=1)
print('Acurácia: {:.2f}'.format(accuracy))