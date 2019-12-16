
import pickle
import numpy as np
import matplotlib.pyplot as plt

BPLT = False

def dist(x,y):
    return (np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2))

def win(pos):
    return dist(pos,(60,60)) < 7

win_indexes = []

shine_win_indexes = []
for k in range(14, 24):
    if k != 18:
        position_record = pickle.load( open( "log_SHINE/position_record_"+str(k)+"_nbgen_100_sizepop_250", "rb" ) )
        for i, pos in enumerate(position_record):
            if win(pos):
                shine_win_indexes.append(i // 250)
                break

win_indexes.append(shine_win_indexes)

me_win_indexes = []
for k in range(1, 11):
    if k != 9:
        position_record = pickle.load( open( "log_MAPelites/position_record_"+str(k)+"_nbgen_100_sizepop_250", "rb" ) )
        for i, pos in enumerate(position_record):
            if win(pos):
                me_win_indexes.append(i // 250)
                break

win_indexes.append(me_win_indexes)

ns_win_indexes = []
for k in range(1, 6):
    position_record = pickle.load( open( "log_NS/position_record_"+str(k)+"_sigma1__nbgen_100_sizepop_250", "rb" ) )
    for i, pos in enumerate(position_record):
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
        plt.plot([sum(1 for e in wi if e < i)/len(wi) for i in range(100)],
        color = c)
    plt.legend(["SHINE", "MAP-Elites", "Novelty search"])

plt.show()
