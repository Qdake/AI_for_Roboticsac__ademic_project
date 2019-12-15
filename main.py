
import gym, gym_fastsim
import time
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
from deap import base
from deap import creator
from deap import tools
import numpy 
import matplotlib.pyplot as plt
from novelty_search import NovArchive
from novelty_search import updateNovelty
import random
from arbre_quaternaire import Quadtree
import pickle
from plot_result import plot
import sys
from simulation import simulation 
from projet_MAPelites_curiosite import MAPelites_curiosite
from projet_MAPelites_ns import MAPelites_ns
from projet_NS import novelty_search
from projet_SHINE import SHINE
########################
##### parametres de l'isntance
#########################
display= False
k = sys.argv[1]  # chaine de caractere pour distinguer differentes instances
nb_generation = int(sys.argv[2])
size_pop = int(sys.argv[3])
algo_name = sys.argv[4]
pb_crossover = 0.1
pb_mutation = 0.9

##################################
#######  main
##################################
# init gym
env = gym.make('FastsimSimpleNavigation-v0')
# type d'algo dispo
assert algo_name in ['SHINE','MAPelites','MAPelites_ns','NS']
# diff cas
if algo_name == 'SHINE':
    position_record, qtree, nb_gen_found = SHINE(env,nb_generation=nb_generation, size_pop=size_pop,pb_crossover=pb_crossover,pb_mutation=pb_mutation,display=display)
if algo_name == 'MAPelites':
    position_record, grid, nb_gen_found = MAPelites_curiosite(env,nb_generation=nb_generation, size_pop=size_pop,pb_crossover=pb_crossover,pb_mutation=pb_mutation,display=display)
if algo_name == 'NS':
    position_record, arc, nb_gen_found = novelty_search(env,nb_generation=nb_generation, size_pop=size_pop,pb_crossover=pb_crossover,pb_mutation=pb_mutation,display=display)
if algo_name == 'MAPelites_ns':
    position_record, grid, nb_gen_found = MAPelites_ns(env,nb_generation=nb_generation, size_pop=size_pop,pb_crossover=pb_crossover,pb_mutation=pb_mutation,display=display)
# fermer gym
env.close()

##########################################
##### sauvegarder le resultat dans un pickle 
##########################################
nfolder = 'log/'+algo_name+'/'
nfile = algo_name + '_' + k +"_gen_" + sys.argv[2] + "_size_" + sys.argv[3]
nimg =  algo_name + '_' + k +"_gen_" + sys.argv[2] + "_size_" + sys.argv[3]
## plot
if algo_name == 'SHINE':
    plot(position_record,nfolder,nimg,qtree)  #plot and save
else:
    plot(position_record,nfolder,nimg,None)
## pickle
data = [position_record,nb_generation]
if algo_name == 'SHINE':
    data.append(qtree)
f = open(nfolder+nfile, 'wb')
pickle.dump(position_record, f)
f.close()