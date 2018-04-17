from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
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
        logging.info("  Resolution: %s", low_res)
        logging.info("  x: %s, y: %s, l: %s", x, y, l)

        timelog = TimeLogger(True)
        data = get_data(low_res, dim, x0=x, y0=y, length=l)
        logging.info("  Done!")
        logging.info(timelog.delta())

        color_sum = sum(sum(data))
        logging.info("  Color Sum: %s on [%s, %s]", color_sum, min_color_ratio*(dim**2), max_color_ratio*(dim**2))

        if color_sum < min_color_ratio*(dim**2) or color_sum > max_color_ratio*(dim**2):
            logging.info("Generated Mandelbrot doesn't meet color requirements.")
            continue

        # low_flat = np.concatenate(data).ravel()
        low_flat = np.expand_dims(data, 2)
        np.savetxt(mb_dir + "/" + file_name, data, fmt='%d')
        lows.append(low_flat)
        logging.info("Generating High-Res Mandelbrot: %s", high_res)
        file_name = "{:.3f}x_{:.3f}y_{:.3f}l_{}res.txt".format(x, y, l, high_res)

        timelog = TimeLogger(True)
        data = get_data(high_res, dim, x0=x, y0=y, length=l)

        high_flat = np.concatenate(data).ravel()
        # high_flat = data

        np.savetxt(mb_dir + "/" + file_name, data, fmt='%d')
        highs.append(high_flat)
        logging.info("  Done!")
        logging.info(timelog.delta())

        i += 1

    x_data = np.array(lows)
    y_data = np.array(highs)

    model = Sequential()

    # Old Model
    # model.add(Dense(256, input_dim=dim**2, kernel_initializer="uniform", activation='relu'))
    # model.add(Dense(256, init='uniform', activation='relu'))
    # model.add(Dense(256, init='uniform', activation='relu'))
    # model.add(Dense(dim**2, init='uniform', activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.add(Conv2D(dim, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=(dim,dim, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(dim**2, activation='hard_sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # model.compile(loss='binary_crossentropy',
    #               optimizer=SGD(lr=0.01),
    #               metrics=['accuracy'])

    model.fit(x_data, y_data, epochs=25, batch_size=25,  verbose=2)

    directory = "models"

    if not os.path.exists(directory):
        os.makedirs(directory)

    model.save("models/" + f"{datetime.datetime.now():%Y-%m-%d_%H:%M%p}" + ".h5")

    return model


low_res = 4
high_res = 16
dim = 32
model = trainer(low_res=low_res, high_res=high_res, dim=dim, count=1000)

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

    # low_test = np.array([np.concatenate(data).ravel()])
    low_test = np.array([np.expand_dims(data, 2)])

    data = get_data(high_res, dim, x0=x, y0=y, length=l)
    # high_test = np.array([np.concatenate(data).ravel()])
    high_test = np.expand_dims(data, 2)

    predictions = model.predict(low_test)

    fig = plt.figure()

    fig.add_subplot(1, 3, 1)
    plt.imshow(np.reshape(low_test, (dim, dim)))

    fig.add_subplot(1, 3, 2)
    plt.imshow(np.reshape(high_test, (dim, dim)))

    fig.add_subplot(1, 3, 3)
    plt.imshow(np.reshape(np.round(predictions[0]), (dim, dim)))

    plt.savefig(directory + '/{}.png'.format(i), bbox_inches='tight')
    plt.close()
