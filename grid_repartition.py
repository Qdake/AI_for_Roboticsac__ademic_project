

import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt

filenames = sys.argv[1:]

for filename in filenames:
    f = pickle.load( open( filename, "rb" ) )

    total_size_x = 600
    total_size_y = 600
    grid_size_x = 60
    grid_size_y = 60
    x_ratio = total_size_x / grid_size_x
    y_ratio = total_size_y / grid_size_y


    grid = np.zeros((grid_size_x,grid_size_y))
    cover = []


    for i in range(len(f)):
        grid[int(f[i][0] // x_ratio)][int(f[i][1] // y_ratio)] = 1
        cover.append(sum(sum(grid)) / (grid_size_x*grid_size_y))
    plt.plot(cover)



print("===== Grid size = " + str(grid_size_x*grid_size_y) + " =====")
plt.show()