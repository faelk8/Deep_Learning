from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense, Activation

iris = load_iris()

# sera usado 4 atributos para previsão
x, y = iris.data[:, :4], iris.target

# dividindo em dados de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.5, random_state=0)

# treinando o modelo
lr = LinearRegression()
lr.fit(x_train, y_train)
pred_y = lr.predict(x_test)

# Definindo o modelo
model = Sequential()
model.add(Dense(16, input_shape=(4,))) # 4 features
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, verbose=1, batch_size=1, epochs=100)

# testando o modelo
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Acurácia: {:.2f}".format(accuracy))

