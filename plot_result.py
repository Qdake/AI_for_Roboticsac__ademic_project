import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def read_img():
    im = Image.open("maze_hard.pbm")    # 读取文件
    im.show()    # 展示图片
    print(im.size)   # 输出图片大小

def plot(position_record,nimg,nfile):
    #=================== Traitement du resultat ==========================================================
    import pickle
    # open a file, where you ant to store the data
    file = open(nimg+nfile, 'wb')     # le 07 nov  X:Y
    # dump information to that file
    pickle.dump(position_record, file)
    # close the file
    file.close()

    # plot
    heatmap = np.zeros((120,120))
    for position in position_record:
        x = int(position[0]//5)
        y = int(position[1]//5)
        heatmap[y][x] += 1
    plt.imshow(heatmap,cmap=plt.cm.hot_r)
    plt.savefig(nimg+nfile)
    plt.show()
