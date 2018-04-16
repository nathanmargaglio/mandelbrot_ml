from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from main import *

seed = 10
np.random.seed(seed)

lows = []
highs = []
dim = 128

for i in range(250):
    x = -1*np.random.random()
    y = np.random.random() - 0.5
    l = np.random.choice([1, 0.5, 0.25, 0.125])

    data = get_data(8, dim, x0=x, y0=y, length=l)

    if sum(sum(data)) < dim:
        continue

    lows.append(np.concatenate(data).ravel())
    data = get_data(dim, dim, x0=x, y0=y, length=l)
    highs.append(np.concatenate(data).ravel())

    print(i)

X = np.array(lows)
Y = np.array(highs)

model = Sequential()
model.add(Dense(128, input_dim=dim**2, init='uniform', activation='relu'))
model.add(Dense(128, init='uniform', activation='relu'))
model.add(Dense(128, init='uniform', activation='relu'))
model.add(Dense(128, init='uniform', activation='relu'))
model.add(Dense(dim**2, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=250, batch_size=10,  verbose=2)

for i in range(100):
    x = -1*np.random.random()
    y = np.random.random() - 0.5
    l = np.random.choice([1, 0.5, 0.25, 0.125])
    data = get_data(8, dim, x0=x, y0=y, length=l)

    if sum(sum(data)) < dim:
        continue

    low_test = np.array([np.concatenate(data).ravel()])

    data = get_data(dim, dim, x0=x, y0=y, length=l)
    high_test = np.array([np.concatenate(data).ravel()])

    predictions = model.predict(low_test)

    fig = plt.figure()

    fig.add_subplot(1, 3, 1)
    plt.imshow(np.reshape(low_test, (dim, dim)))

    fig.add_subplot(1, 3, 2)
    plt.imshow(np.reshape(high_test, (dim, dim)))

    fig.add_subplot(1, 3, 3)
    plt.imshow(np.reshape(np.round(predictions[0]), (dim, dim)))

    plt.show()
