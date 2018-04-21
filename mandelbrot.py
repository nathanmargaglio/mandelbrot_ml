import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import logging
from utils import TimeLogger
import os
import re

np.set_printoptions(threshold=np.inf)
logging.basicConfig(format='%(asctime)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)


def f(z, c):
    return z ** 2 + c


def bounded(c, loop_limit=100, stop_limit=100):
    value = f(0, c)
    for i in range(loop_limit):
        value = f(value, c)
        if abs(value) > stop_limit:
            return 0.
    return 1.


def generate(x0=0., y0=0., length=2., dim=64, res=1):
    lower_x = x0 - length
    upper_x = x0 + length
    lower_y = y0 - length
    upper_y = y0 + length

    x_r = np.linspace(lower_x, upper_x, dim)
    y_r = np.linspace(lower_y, upper_y, dim)

    results = []
    for x in x_r:
        row = []
        for y in y_r:
            mb = bounded(x + y * 1j)
            for r in range(res):
                row.append(mb)
        for r in range(res):
            results.append(row)

    return np.array(results, dtype=float).T


def sample_zoom_animation():
    fig = plt.figure()

    ims = []
    for l in np.linspace(2, 0.025, 30):
        mb = generate(x0=-0.75, dim=128, length=l)
        res = np.array(mb, dtype=int).T
        im = plt.imshow(res, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
    plt.show()


def sample_res_animation():
    fig = plt.figure()

    ims = []

    factors = [2 ** d for d in range(4, 10)]
    for d in factors:
        mb = generate(dim=d, res=int(max(factors) / d))
        res = np.array(mb, dtype=int).T
        im = plt.imshow(res, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
    plt.show()


def get_data(d, dh, x0=-0.5, y0=0., length=1.):
    return generate(dim=d, x0=x0, y0=y0, length=length, res=int(dh / d))


def generate_data_sets(low_res=16, high_res=32, dim=32, radii=None, count=100,
                       min_color_ratio=0.05, max_color_ratio=0.95,
                       prep='conv', load_available=True):
    lows = []
    highs = []

    if not radii:
        radii = [1.0, 0.75, 0.5, 0.25]

    mb_dir = "mandelbrots/{}".format(dim)

    if not os.path.exists(mb_dir):
        os.makedirs(mb_dir)

    available_data = os.listdir(mb_dir)
    i = 0
    while i < count:

        # Low-Res Mandelbrot
        low_res_files = [x for x in available_data if "{}res".format(low_res) in x]
        if load_available and len(low_res_files):
            file = np.random.choice(low_res_files)
            available_data.remove(file)
            variables = re.search("(.+)x_(.+)y_(.+)l_(.+)res.txt", file)
            x = float(variables.group(1))
            y = float(variables.group(2))
            l = float(variables.group(3))
        else:
            logging.info("Generating %sx%s Mandelbrot Data: %s of %s", dim, dim, i, count)
            x = round(-1 * np.random.random(), 2)
            y = round(np.random.random() - 0.5, 2)
            l = np.random.choice(radii)

        file_name = mb_dir + "/" + "{:.3f}x_{:.3f}y_{:.3f}l_{}res.txt".format(x, y, l, low_res)
        if os.path.isfile(file_name):
            logging.info("Loading Low-Res Mandelbrot: %s", low_res)
            logging.info("  Resolution: %s", low_res)
            logging.info("  x: %s, y: %s, l: %s", x, y, l)
            low_data = np.loadtxt(file_name)
        else:
            logging.info("Generating Low-Res Mandelbrot: %s", low_res)
            logging.info("  Resolution: %s", low_res)
            logging.info("  x: %s, y: %s, l: %s", x, y, l)

            timelog = TimeLogger(True)
            low_data = get_data(low_res, dim, x0=x, y0=y, length=l)

            color_sum = sum(sum(low_data))
            logging.info("  Color Sum: %s on [%s, %s]", color_sum, min_color_ratio * (dim ** 2),
                         max_color_ratio * (dim ** 2))

            if color_sum < min_color_ratio * (dim ** 2) or color_sum > max_color_ratio * (dim ** 2):
                logging.info("Generated Mandelbrot doesn't meet color requirements.")
                continue

            np.savetxt(file_name, low_data, fmt='%d')
            logging.info("  Done!")
            logging.info(timelog.delta())

        # High-Res Mandelbrot

        file_name = mb_dir + "/" + "{:.3f}x_{:.3f}y_{:.3f}l_{}res.txt".format(x, y, l, high_res)
        if os.path.isfile(file_name):
            logging.info("Loading High-Res Mandelbrot: %s", low_res)
            logging.info("  Resolution: %s", low_res)
            logging.info("  x: %s, y: %s, l: %s", x, y, l)
            high_data = np.loadtxt(file_name)
        else:
            logging.info("Generating High-Res Mandelbrot: %s", high_res)
            timelog = TimeLogger(True)
            high_data = get_data(high_res, dim, x0=x, y0=y, length=l)
            np.savetxt(file_name, high_data, fmt='%d')
            logging.info("  Done!")
            logging.info(timelog.delta())

        if prep == 'conv':
            low_flat = np.expand_dims(low_data, 2)
            high_flat = np.expand_dims(high_data, 2)
        elif prep == 'flat':
            low_flat = np.concatenate(low_data).ravel()
            high_flat = np.concatenate(high_data).ravel()
        else:
            low_flat = low_data
            high_flat = high_data

        lows.append(low_flat)
        highs.append(high_flat)

        i += 1

    x_data = np.array(lows)
    y_data = np.array(highs)

    return x_data, y_data


if __name__ == "__main__":
    fig = plt.figure()

    ims = []

    factors = [8, 32, 128]
    for d in factors:
        mb = generate(dim=d, x0=-0.5, length=1., res=int(max(factors)/d))
        res = np.array(mb, dtype=float).T
        im = plt.imshow(res, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
    plt.show()

    #print(res)
    #plt.imshow(res)
    #plt.show()
