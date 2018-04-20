import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

np.set_printoptions(threshold=np.inf)


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
    return results


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
    mb = generate(dim=d, x0=x0, y0=y0, length=length, res=int(dh / d))
    return np.array(mb, dtype=float).T


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
