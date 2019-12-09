import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

def read_img():
    im = Image.open("maze_hard.pbm")    # 读取文件
    im.show()    # 展示图片
    print(im.size)   # 输出图片大小

def plot(position_record,nfolder,nimg,qtree):
    #=================== Traitement du resultat ==========================================================
    # open a file, where you ant to store the data

    
    if qtree:
        qtree.plot()
    
    # plot
    x = [position[0] for position in position_record]
    y = [600-position[1] for position in position_record]

    plt.scatter(x,y,color="b",marker=".")
    """
    for position in position_record:
        x = int(position[0]//5)
        y = int(position[1]//5)
        heatmap[y][x] += 1
    """


    #plt.imshow(heatmap,cmap=plt.cm.hot_r)
    plt.savefig(nfolder+nimg)
    plt.show()
    
