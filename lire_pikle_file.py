#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 19:28:10 2019

@author: qiu-adm
"""
import numpy as np
import pickle
from deap import base
from deap import creator
from plot_result import plot_position_record
from plot_result import plot_qtree
## creat class Individual
IND_SIZE = 192 #(5,2,2,10)
#create class
creator.create("FitnessMax",base.Fitness,weights=(1.0,))
creator.create("Individual",list,fitness=creator.FitnessMax,pos=list,profondeur=float)
# data
nfolder = 'log/MAPelites/'
#nfolder = 'log/MAPelites_ns/'
#nfolder = 'log/NS/'
nfile = 'MAPelites_0_gen_1000_size_250MAPelites'
#data = pickle.load( open( "log/MAPelites/MAPelites_0_gen_1000_size_250", "rb" ) )
data = pickle.load( open( nfolder+nfile, "rb" ) )

# position_record
position_record = data[0]

# nb_gen_found
nb_gen_found = data[1]

# qtree
if len(data)==3:
    qtree = data[2]

#qtree = pickle.load( open( "log_SHINE/tree_record_121201_nbgen_300_sizepop_250", "rb" ) )

plot_position_record(position_record,nfolder,nfile)
if 'qtree' in dir(): # si qtree est definie
    plot_qtree(qtree,nfolder,nfile)

for pos in position_record:
    if np.sqrt((pos[0]-60)**2+(pos[1]-60)**2)<=7:
        print(pos)

print("but atteint ? nb_generation = ",nb_gen_found)

