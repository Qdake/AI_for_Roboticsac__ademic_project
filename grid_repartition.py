

import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

filenames = sys.argv[1:]
colors = ['#bb2288', '#22cc88', '#229977']


for k, filename in enumerate(filenames):

    f = pickle.load( open( filename, "rb" ) )

    total_size_x = 600
    total_size_y = 600
    grid_size_x = 60
    grid_size_y = 60
    x_ratio = total_size_x / grid_size_x
    y_ratio = total_size_y / grid_size_y

    if len(f) == 2:
        f = f[0]

    grid = np.zeros((grid_size_x,grid_size_y))
    cover = []

    for i in range(len(f)):
        grid[int(f[i][0] // x_ratio)][int(f[i][1] // y_ratio)] = 1
        if i%250 == 0:
            cover.append(sum(sum(grid)) / (grid_size_x*grid_size_y))

    plt.plot(cover[:350], color=colors[k])

legend = ["SHINE", "MAP-Elites", "Novelty search"]

plt.legend(legend)
plt.show()