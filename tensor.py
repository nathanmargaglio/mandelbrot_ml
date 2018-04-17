from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from main import *
import logging
from utils import TimeLogger
import os
import datetime


logging.basicConfig(format='%(asctime)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)
seed = 10
np.random.seed(seed)


def trainer(low_res=2, high_res=32, dim=32, radii=None, count=100, min_color_ratio=0.25, max_color_ratio=0.75):
    lows = []
    highs = []

    if not radii:
        radii = [1., 0.5, 0.25, 0.125]

    mb_dir = "mandelbrots/{}".format(dim)

    if not os.path.exists(mb_dir):
        os.makedirs(mb_dir)

    i = 0
    while i < count:
        logging.info("Generating %sx%s Mandelbrot Data: %s of %s", dim, dim, i, count)
        x = round(-1*np.random.random(), 3)
        y = round(np.random.random() - 0.5, 3)
        l = np.random.choice(radii)

        file_name = "{:.3f}x_{:.3f}y_{:.3f}l_{}res.txt".format(x, y, l, low_res)

        logging.info("Generating Low-Res Mandelbrot: %s", low_res)
        logging.info("Resolution: %s", low_res)
        logging.info("x: %s, y: %s, l: %s", x, y, l)

        timelog = TimeLogger(True)
        data = get_data(low_res, dim, x0=x, y0=y, length=l)
        logging.info("Done!")
        logging.info(timelog.delta())

        color_sum = sum(sum(data))
        logging.info("Color Sum: %s on [%s, %s]", color_sum, min_color_ratio*(dim**2), max_color_ratio*(dim**2))

        if color_sum < min_color_ratio*(dim**2) or color_sum > max_color_ratio*(dim**2):
            logging.info("Generated Mandelbrot doesn't meet color requirements.")
            continue

        low_flat = np.concatenate(data).ravel()
        np.savetxt(mb_dir + "/" + file_name, low_flat, fmt='%d')
        lows.append(low_flat)
        logging.info("Generating High-Res Mandelbrot: %s", high_res)
        file_name = "{:.3f}x_{:.3f}y_{:.3f}l_{}res.txt".format(x, y, l, high_res)

        timelog = TimeLogger(True)
        data = get_data(high_res, dim, x0=x, y0=y, length=l)
        high_flat = np.concatenate(data).ravel()

        np.savetxt(mb_dir + "/" + file_name, high_flat, fmt='%d')
        highs.append(high_flat)
        logging.info("Done!")
        logging.info(timelog.delta())

        i += 1

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


low_res = 2
high_res = 64
dim = 256
model = trainer(low_res=low_res, high_res=high_res, dim=dim, count=100)

directory = "results/res_" + f"{datetime.datetime.now():%Y-%m-%d_%H:%M%p}"

if not os.path.exists(directory):
    os.makedirs(directory)

for i in range(50):
    x = -1*np.random.random()
    y = np.random.random() - 0.5
    l = np.random.choice([1, 0.5, 0.25, 0.125])
    data = get_data(low_res, dim, x0=x, y0=y, length=l)

    if sum(sum(data)) < dim:
        continue

    low_test = np.array([np.concatenate(data).ravel()])

    data = get_data(high_res, dim, x0=x, y0=y, length=l)
    high_test = np.array([np.concatenate(data).ravel()])

    predictions = model.predict(low_test)

    fig = plt.figure()

    fig.add_subplot(1, 3, 1)
    plt.imshow(np.reshape(low_test, (dim, dim)))

    fig.add_subplot(1, 3, 2)
    plt.imshow(np.reshape(high_test, (dim, dim)))

    fig.add_subplot(1, 3, 3)
    plt.imshow(np.reshape(np.round(predictions[0]), (dim, dim)))

    plt.savefig(directory + '/{}.png'.format(i), bbox_inches='tight')
    plt.clf()
