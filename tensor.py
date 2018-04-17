from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from main import *
import logging
from utils import TimeLogger


logging.basicConfig(format='%(asctime)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)
seed = 10
np.random.seed(seed)


def trainer(res=2, dim=32, radii=None, count=100, min_color_sum=0, max_color_sum=None):
    lows = []
    highs = []
    color_sum = None

    if not radii:
        radii = [1., 0.5, 0.25, 0.125]

    if not max_color_sum:
        max_color_sum = dim**2

    assert min_color_sum >= 0 and max_color_sum <= dim**2

    for i in range(count):
        logging.info("Generating %sx%s Mandelbrot Data: %s of %s", dim, dim, i, count)
        x = -1*np.random.random()
        y = np.random.random() - 0.5
        l = np.random.choice(radii)

        logging.info("Generating Low-Res Mandelbrot: %s", res)
        logging.info("Resolution: %s", res)
        logging.info("x: %s, y: %s, l: %s", x, y, l)

        timelog = TimeLogger(True)
        data = get_data(res, dim, x0=x, y0=y, length=l)
        logging.info("Done!")
        logging.info(timelog.delta())

        color_sum = sum(sum(data))
        logging.info("Color Sum: %s on [%s, %s]", color_sum, min_color_sum, max_color_sum)

        lows.append(np.concatenate(data).ravel())
        logging.info("Generating High-Res Mandelbrot: %s", dim)
        timelog = TimeLogger(True)
        data = get_data(dim, dim, x0=x, y0=y, length=l)
        highs.append(np.concatenate(data).ravel())
        logging.info("Done!")
        logging.info(timelog.delta())

    x_data = np.array(lows)
    y_data = np.array(highs)

    model = Sequential()
    model.add(Dense(256, input_dim=dim**2, kernel_initializer="uniform", activation='relu'))
    model.add(Dense(256, init='uniform', activation='relu'))
    model.add(Dense(256, init='uniform', activation='relu'))
    model.add(Dense(256, init='uniform', activation='relu'))
    model.add(Dense(256, init='uniform', activation='relu'))
    model.add(Dense(256, init='uniform', activation='relu'))
    model.add(Dense(dim**2, init='uniform', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_data, y_data, epochs=200, batch_size=10,  verbose=2)

    return model


res = 16
dim = 256
model = trainer(dim=dim, count=1000)

for i in range(50):
    x = -1*np.random.random()
    y = np.random.random() - 0.5
    l = np.random.choice([1, 0.5, 0.25, 0.125])
    data = get_data(res, dim, x0=x, y0=y, length=l)

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

    plt.savefig('results/{}.png'.format(i), bbox_inches='tight')
    plt.clf()
