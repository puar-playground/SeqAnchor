import numpy as np
import matplotlib.pyplot as plt
import imageio
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def rainbow_cmap(n_color):
    cdict = {'red': ((0.0, 0.0, 0.0),
                     (0.1, 0.5, 0.5),
                     (0.2, 0.0, 0.0),
                     (0.4, 0.2, 0.2),
                     (0.6, 0.0, 0.0),
                     (0.8, 1.0, 1.0),
                     (1.0, 1.0, 1.0)),
             'green': ((0.0, 0.0, 0.0),
                       (0.1, 0.0, 0.0),
                       (0.2, 0.0, 0.0),
                       (0.4, 1.0, 1.0),
                       (0.6, 1.0, 1.0),
                       (0.8, 1.0, 1.0),
                       (1.0, 0.0, 0.0)),
             'blue': ((0.0, 0.0, 0.0),
                      (0.1, 0.5, 0.5),
                      (0.2, 1.0, 1.0),
                      (0.4, 1.0, 1.0),
                      (0.6, 0.0, 0.0),
                      (0.8, 0.0, 0.0),
                      (1.0, 0.0, 0.0))}

    colormap = matplotlib.colors.LinearSegmentedColormap('rainbow', cdict, n_color)
    colormap = colormap.reversed()
    return colormap

if __name__ == "__main__":

    rain_bow = rainbow_cmap(20)

    plt.figure(figsize=(20, 10))
    for i in range(20):
        a = np.zeros([20])
        x = np.array(list(range(20)))
        a[i] = 1
        plt.bar(x, a, color=rain_bow(i))

    plt.savefig('my_cmap.png')
