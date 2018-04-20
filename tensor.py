from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Reshape, MaxPooling2D, Flatten, Conv2DTranspose
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from mandelbrot import *
import logging
from utils import TimeLogger
import os
import sys
import datetime


logging.basicConfig(format='%(asctime)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)
seed = 10
np.random.seed(seed)


def generate_data_sets(low_res=2, high_res=32, dim=32, radii=None, count=100, min_color_ratio=0.05, max_color_ratio=0.95):
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
        x = round(-1 * np.random.random(), 3)
        y = round(np.random.random() - 0.5, 3)
        l = np.random.choice(radii)

        x = 0
        y = 0
        l = 1.5

        file_name = "{:.3f}x_{:.3f}y_{:.3f}l_{}res.txt".format(x, y, l, low_res)

        logging.info("Generating Low-Res Mandelbrot: %s", low_res)
        logging.info("  Resolution: %s", low_res)
        logging.info("  x: %s, y: %s, l: %s", x, y, l)

        timelog = TimeLogger(True)
        data = get_data(low_res, dim, x0=x, y0=y, length=l)
        logging.info("  Done!")
        logging.info(timelog.delta())

        color_sum = sum(sum(data))
        logging.info("  Color Sum: %s on [%s, %s]", color_sum, min_color_ratio * (dim ** 2),
                     max_color_ratio * (dim ** 2))

        # if color_sum < min_color_ratio * (dim ** 2) or color_sum > max_color_ratio * (dim ** 2):
        #     logging.info("Generated Mandelbrot doesn't meet color requirements.")
        #     continue

        # low_flat = np.concatenate(data).ravel()
        low_flat = np.expand_dims(data, 2)
        np.savetxt(mb_dir + "/" + file_name, data, fmt='%d')
        lows.append(low_flat)
        logging.info("Generating High-Res Mandelbrot: %s", high_res)
        file_name = "{:.3f}x_{:.3f}y_{:.3f}l_{}res.txt".format(x, y, l, high_res)

        timelog = TimeLogger(True)
        data = get_data(high_res, dim, x0=x, y0=y, length=l)

        high_flat = np.expand_dims(data, 2)
        # high_flat = np.concatenate(data).ravel()
        # high_flat = data

        np.savetxt(mb_dir + "/" + file_name, data, fmt='%d')
        highs.append(high_flat)
        logging.info("  Done!")
        logging.info(timelog.delta())

        i += 1

    x_data = np.array(lows)
    y_data = np.array(highs)

    return x_data, y_data


def trainer(x_data, y_data):

    dim = x_data.shape[1]
    model = Sequential()

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

    model.fit(x_data, y_data, epochs=25, batch_size=10, validation_split=0.2)

    directory = "models"

    if not os.path.exists(directory):
        os.makedirs(directory)

    model.save("models/" + f"{datetime.datetime.now():%Y-%m-%d_%H:%M%p}" + ".h5")

    return model


def SRCNN(x_data, y_data, load_data=True):
    dim = x_data.shape[1]

    
    model = Sequential()
    adam = Adam(lr=0.001, decay=0.0001)

    model.add(Conv2D(filters=256, kernel_size=(4,4), padding='valid', input_shape=(dim, dim, 1),
                     activation='relu', init='he_normal', use_bias=True))
    #model.add(Conv2D(filters=128, kernel_size=(8,8), padding='valid', init='he_normal', use_bias=True))
    #model.add(Dense(1000, activation='sigmoid', use_bias=True))
    model.add(Dense(1000, activation='sigmoid', use_bias=True))
    #model.add(Conv2DTranspose(filters=128, kernel_size=(8, 8), kernel_initializer='glorot_uniform',
    #                          activation='sigmoid', padding='valid', use_bias=True))
    model.add(Conv2DTranspose(filters=1, kernel_size=(4, 4), kernel_initializer='glorot_uniform',
                       activation='linear', padding='valid', use_bias=True))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    # checkpoint
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, TensorBoard(histogram_freq=0, write_images=True)]

    if load_data:
        model.load_weights(filepath)
    else:
        model.fit(x_data, y_data, epochs=25, batch_size=10, validation_split=0.2,
              callbacks=callbacks_list)
    return model


def test_model(low_res, high_res, model):
    directory = "results/res_" + f"{datetime.datetime.now():%Y-%m-%d_%H:%M%p}"

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(1):
        x = -1 * np.random.random()
        y = np.random.random() - 0.5
        l = np.random.choice([1, 0.5, 0.25, 0.125])

        x = 0
        y = 0
        l = 2.0

        data = get_data(low_res, dim, x0=x, y0=y, length=l)
        low_test = np.array([np.expand_dims(data, 2)])

        data = get_data(high_res, dim, x0=x, y0=y, length=l)
        # high_test = np.expand_dims(data, 2)
        high_test = np.array([np.expand_dims(data, 2)])

        predictions = model.predict(low_test)

        fig = plt.figure()

        fig.add_subplot(1, 3, 1)
        plt.imshow(np.reshape(low_test, (dim, dim)))

        fig.add_subplot(1, 3, 2)
        plt.imshow(np.reshape(high_test, (dim, dim)))

        fig.add_subplot(1, 3, 3)
        plt.imshow(np.reshape(predictions[0], (dim, dim)))

        plt.savefig(directory + '/{}.png'.format(i), bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    low_res = 16
    high_res = 64
    dim = 64
    count = 1
    load_model = False
    #load_model = True

    if len(sys.argv) > 1:
        load_model = (sys.argv[1] == 'true')

    if not load_model:
        x_data, y_data = generate_data_sets(low_res=low_res, high_res=high_res, dim=dim, count=count)
    else:
        x_data, y_data = generate_data_sets(low_res=low_res, high_res=high_res, dim=dim, count=2)

    sam_x = x_data[0]
    sam_y = y_data[0]

    x_data = np.array([sam_x]*100)
    y_data = np.array([sam_y]*100)
    model = SRCNN(x_data, y_data, load_model)
    test_model(low_res, high_res, model)
