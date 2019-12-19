import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import pickle
from deap import base
from deap import creator
## creat class Individual
IND_SIZE = 192 #(5,2,2,10)
#create class
creator.create("FitnessMax",base.Fitness,weights=(1.0,))
creator.create("Individual",list,fitness=creator.FitnessMax,pos=list,profondeur=float)


from PIL import Image

def read_img():
    im = Image.open("maze_hard.pbm")
    im.show()
    print(im.size)

def plot_qtree(qtree,nfolder,nimg):
    qtree.plot()
    plt.savefig(nfolder+nimg)
    #plt.show()

def plot_position_record(position_record,nfolder,nimg):
    #=================== Traitement du resultat ==========================================================
    # open a file, where you ant to store the data

    img = mpimg.imread('maze_hard.pbm','PBM')
    plt.imshow(img)

    # plot
    x = [int(position[0]) for position in position_record]
    y = [int(position[1]) for position in position_record]

    heatmap = np.zeros((120,120))
    for position in position_record:
        i = int(position[0]//5)
        j = int(position[1]//5)
        heatmap[i][j] += 1

    m = np.amax(heatmap)
    heatmap = pow(heatmap/float(m), 1/3)


    new_x = []
    new_y = []
    couples = []
    for xi, yi in zip(x, y):
        if not ((xi//5, yi//5) in couples):
            new_x.append(xi)
            new_y.append(yi)
            couples.append((xi//5,yi//5))

    colors = []
    for xi, yi in zip(new_x,new_y):
        colors.append((heatmap[xi//5][yi//5], 0, 0.3))

    plt.scatter(new_x,new_y,color=colors,marker=".")

    plt.savefig(nfolder+nimg)
    #plt.show()

# TO REMOVE
if __name__ == "__main__":
    path= "log/MAPelites/"
    file="MAPelites_genFix_400_size_250"
    with open( path+file, "rb" ) as f:
        pr = pickle.load(f)
        plot_position_record(pr[0], path, file) 