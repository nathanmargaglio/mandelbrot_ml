from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Reshape, MaxPooling2D, Flatten, Conv2DTranspose, BatchNormalization
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
        radii = [0.5]

    mb_dir = "mandelbrots/{}".format(dim)

    if not os.path.exists(mb_dir):
        os.makedirs(mb_dir)

    i = 0
    while i < count:
        logging.info("Generating %sx%s Mandelbrot Data: %s of %s", dim, dim, i, count)
        x = round(-1 * np.random.random(), 3)
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

        # high_flat = np.concatenate(data).ravel()
        # high_flat = data
        high_flat = np.expand_dims(data, 2)

        np.savetxt(mb_dir + "/" + file_name, data, fmt='%d')
        highs.append(high_flat)
        logging.info("  Done!")
        logging.info(timelog.delta())

        i += 1

    x_data = np.array(lows)
    y_data = np.array(highs)

    return x_data, y_data


def SRCNN(x_data, y_data, load_data=True):
    shape = x_data[0].shape
    
    model = Sequential()
    opt = Adam(lr=0.001, decay=0.0001)
    # opt = SGD(lr=0.1, decay=0.001, momentum=0.99, nesterov=True)

    # model.add(Dense(1000, activation='tanh', use_bias=True, input_shape=dim))
    # model.add(BatchNormalization())
    # model.add(Dense(1000, activation='tanh', use_bias=True))
    # model.add(BatchNormalization())
    # model.add(Dense(1000, activation='tanh', use_bias=True))
    # model.add(BatchNormalization())
    # model.add(Dense(1000, activation='tanh', use_bias=True))
    # model.add(BatchNormalization())
    # model.add(Dense(dim[0], activation='linear', use_bias=True))
    # model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=(1, 1), input_shape=shape))
    model.add(Dense(256, activation='sigmoid', use_bias=True))
    model.add(Dense(256, activation='sigmoid', use_bias=True))
    model.add(Conv2DTranspose(1, kernel_size=(1, 1), activation='linear', use_bias=True))

    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

    # checkpoint
    filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint, TensorBoard(histogram_freq=0, write_images=True)]

    if load_data:
        model.load_weights(filepath)
    else:
        print(x_data.shape)
        model.fit(x_data, y_data, epochs=250, batch_size=10,
              callbacks=callbacks_list)
    return model


def test_model(low_res, high_res, model, show=False):
    directory = "results/res_" + f"{datetime.datetime.now():%Y-%m-%d_%H:%M%p}"

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(50):
        x = -1 * np.random.random()
        y = np.random.random() - 0.5
        l = np.random.choice([1, 0.5, 0.25, 0.125])

        data = get_data(low_res, dim, x0=x, y0=y, length=l)
        low_test = np.array([np.expand_dims(data, 2)])
        # low_test = np.array([np.concatenate(data).ravel()])

        data = get_data(high_res, dim, x0=x, y0=y, length=l)
        high_test = np.array([np.expand_dims(data, 2)])
        # high_test = np.array([np.concatenate(data).ravel()])

        predictions = model.predict(low_test)

        fig = plt.figure()

        fig.add_subplot(1, 3, 1)
        low_fig = np.reshape(low_test, (dim, dim))
        plt.imshow(low_fig)

        fig.add_subplot(1, 3, 2)
        high_fig = np.reshape(high_test, (dim, dim))
        plt.imshow(high_fig)

        fig.add_subplot(1, 3, 3)
        pred_fig = np.reshape(predictions[0], (dim, dim))
        plt.imshow(pred_fig)

        plt.savefig(directory + '/{}.png'.format(i), bbox_inches='tight')
        if show:
            print("LOW_MAX", np.amax(low_fig))
            print("LOW_MIN", np.amin(low_fig))

            print("HIG_MAX", np.amax(high_fig))
            print("HIG_MIN", np.amin(high_fig))

            print("PRE_MAX", np.amax(pred_fig))
            print("PRE_MIN", np.amin(pred_fig))
            plt.show()
        plt.close()


if __name__ == "__main__":
    low_res = 16
    high_res = 64
    dim = 64
    count = 10
    load_model = False
    #load_model = True

    if len(sys.argv) > 1:
        load_model = (sys.argv[1] == 'load')

    if not load_model:
        x_data, y_data = generate_data_sets(low_res=low_res, high_res=high_res, dim=dim, count=count)
    else:
        x_data, y_data = generate_data_sets(low_res=low_res, high_res=high_res, dim=dim, count=1)

    model = SRCNN(x_data, y_data, load_model)
    test_model(low_res, high_res, model, show=True)
