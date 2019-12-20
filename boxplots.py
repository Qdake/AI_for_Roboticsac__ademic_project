
import pickle
import numpy as np
import matplotlib.pyplot as plt

from deap import base
from deap import creator

## creat class Individual
IND_SIZE = 192 #(5,2,2,10)
#create class
creator.create("FitnessMax",base.Fitness,weights=(1.0,))
creator.create("Individual",list,fitness=creator.FitnessMax,pos=list,profondeur=float)

BPLT = True

def dist(x,y):
    return (np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2))

def win(pos):
    return dist(pos,(60,60)) < 7

win_indexes = []
# SHINE
shine_win_indexes = []
for k in range(0, 16):
    if k != 18:
        position_record = pickle.load( open( "log/SHINE/SHINE_"+str(k)+"_gen_1000_size_250", "rb" ) )
        for i, pos in enumerate(position_record[0]):
            if win(pos):
                shine_win_indexes.append(i // 250)
                break

win_indexes.append(shine_win_indexes)

# MAP elites
me_win_indexes = []
for k in range(1, 20):
    if k != 9:
        position_record = pickle.load( open( "log/MAPelites/MAPelites_"+str(k)+"_gen_1000_size_250", "rb" ) )
        for i, pos in enumerate(position_record[0]):
            if win(pos):
                me_win_indexes.append(i // 250)
                break

win_indexes.append(me_win_indexes)

# NS
ns_win_indexes = [] #[699,616,82,461,426,372]

for k in range(0,16):
    position_record = pickle.load( open( "log/NS/NS_"+str(k)+"_gen_1000_size_250", "rb" ) )
    for i, pos in enumerate(position_record[0]):
        if win(pos):
            ns_win_indexes.append(i // 250)
            break

win_indexes.append(ns_win_indexes)

colors = ['#bb2288', '#22cc88', '#229977']

if BPLT:
    bplot = plt.boxplot(win_indexes, labels=["SHINE", "MAP-Elites", "Novelty search"], patch_artist=True)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
else:
    for wi, c in zip(win_indexes, colors):
        plt.plot([sum(1 for e in wi if e < i)/len(wi) for i in range(600)],
        color = c)
    plt.legend(["SHINE", "MAP-Elites", "Novelty search"])

plt.savefig("analyse/couverture")
plt.show()
