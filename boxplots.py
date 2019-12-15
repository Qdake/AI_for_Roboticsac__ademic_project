
import pickle
import numpy as np

def dist(x,y):
    return (np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2))

def win(pos):
    return dist(pos,(60,60)) < 7

position_record = pickle.load( open( "log_SHINE/position_record_121201_nbgen_300_sizepop_250", "rb" ) )


for i, pos in enumerate(position_record):
    if win(pos):
        print(i // 250)
        break