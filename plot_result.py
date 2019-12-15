import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

from PIL import Image

def read_img():
    im = Image.open("maze_hard.pbm")    
    im.show()    
    print(im.size)   
def plot_qtree(qtree,nfolder,nimg):
    qtree.plot()
    plt.savefig(nfolder+nimg)
    plt.show()

def plot_position_record(position_record,nfolder,nimg):
    #=================== Traitement du resultat ==========================================================
    # open a file, where you ant to store the data

    img = mpimg.imread('maze_hard.pbm','PBM')
    plt.imshow(img)
    
    # plot
    x = [position[0] for position in position_record]
    y = [position[1] for position in position_record]
    #y = [600-position[1] for position in position_record]

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
    
