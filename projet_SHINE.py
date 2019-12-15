#!/usr/bin/python
# -*- coding: utf-8 -*-

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

def choix_a_roulette(population_list, size_pop):
    profondeurs = [ind.profondeur for ind in population_list]
    distribution = [1/pow(4,profondeur) for profondeur in profondeurs]
    somme = sum(distribution)
    distribution = [i/somme for i in distribution]
    #print("population list    ***  ", population_list)
    indices = np.random.choice(list(range(len(population_list))),size_pop,replace = True,p=distribution)
    return [population_list[i] for i in indices]
def dist(x,y):
    return (np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2))

def SHINE(env,size_pop=50,pb_crossover=0.6, pb_mutation=0.3, nb_generation=100, display=False):

    IND_SIZE = 192 #(5,2,2,10)
    random.seed()
    nb_gen_found = nb_generation

    #create class
    creator.create("FitnessMax",base.Fitness,weights=(1.0,))
    creator.create("Individual",list,fitness=creator.FitnessMax,pos=list,profondeur=float)
    # toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.normal)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list,  toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate",tools.cxBlend,alpha=0.1)

    # initialisation
    population_list = list()
    arbre = Quadtree(0,600,0,600)
    # pour plot heatmap
    position_record = []
    populations = []

    # generer la population initiale
    pop = toolbox.population(size_pop)
    # simulation
    for ind in pop:
        ind.bd,but_atteint = simulation(env,ind,display=display)
        position_record.append(ind.bd)
        succes = arbre.ajout(ind)
        if succes:
            population_list.append(ind)

    for gen in range(1, nb_generation+1):
        print("generation ",gen)

        # Select the next generation individuals
        pop = choix_a_roulette(population_list,size_pop)    # population est l'ensemble des individus qui presentent dans le grid
        # Clone the selected individuals
        pop = list(map(toolbox.clone, pop))
        populations.append(pop)


        # crossover
        for child1, child2 in zip(pop[::2], pop[1::2]):
            if np.random.random()<pb_crossover:
                toolbox.mate(child1,child2)

        #mutation
        for mutant in pop:
            if np.random.random() < pb_mutation:
                tools.mutGaussian(mutant, mu=0.0, sigma=1, indpb=0.1)

        # simulation
        invalid_inds = [ind for ind in pop if ind.fitness.valid == False]
        for ind in invalid_inds:
            ind.bd,but_atteint = simulation(env,ind,display=display)
            position_record.append(ind.bd)
            if arbre.ajout(ind):
                population_list.append(ind)
            if but_atteint:
                print("===== but atteint =====================================")
                print(gen)
                print("==========================================")
                return position_record, arbre, gen

            
    return position_record, arbre, None





if __name__ == "__main__":

    st = time.time()

    display= False
    nb_generation = int(sys.argv[2])
    size_pop = int(sys.argv[3])
    #simulation(env,None,True)
    #env.close()
    k = sys.argv[1]
    nfolder = 'log_SHINE/'
        
    env = gym.make('FastsimSimpleNavigation-v0')
    position_record,qtree,nb_gen_found = es(env,nb_generation=nb_generation, size_pop=size_pop,pb_crossover=0.1,pb_mutation=0.9,display=display)
    env.close()

    nfolder = 'log_SHINE/'
    nfile = 'position_record_' + k +"_nbgen_"+sys.argv[2]+"_sizepop_"+sys.argv[3]
    nimg = 'position_record_' + k +"_nbgen_"+sys.argv[2]+"_sizepop_"+sys.argv[3]
    ntree = 'tree_record_' + k +"_nbgen_"+sys.argv[2]+"_sizepop_"+sys.argv[3]
    plot(position_record,nfolder,nimg,qtree)  #plot and save

    #qtree.plot()
    #plt.show()
    #plot(position_record,nfolder,nimg,qtree)  #plot and save

    f = open(nfolder+nfile, 'wb')
    pickle.dump(position_record, f)
    f.close()

    f = open(nfolder+ntree, 'wb') 
    pickle.dump(qtree, f)
    f.close()

    nb_gen_found_log.append(nb_gen_found)

    print(nb_gen_found_log)

    #qtree.plot()
    #plt.show()

    f = open(nfolder+'nb_gen_found'+k, 'wb')
    pickle.dump(nb_gen_found_log, f)
    f.close






    #=================== Traitement du resultat ==========================================================
    """
    k = sys.argv[1]
    nfolder = 'log_SHINE/'
    nfile = 'position_record_' + k +"_nbgen_"+sys.argv[2]+"_sizepop_"+sys.argv[3]
    nimg = 'position_record_' + k +"_nbgen_"+sys.argv[2]+"_sizepop_"+sys.argv[3]
    ntree = 'tree_record_' + k +"_nbgen_"+sys.argv[2]+"_sizepop_"+sys.argv[3]
    plot(position_record,nfolder,nimg,qtree)  #plot and save

    f = open(nfolder+nfile, 'wb')
    pickle.dump(position_record, f)
    f.close()

    f = open(nfolder+ntree, 'wb') 
    pickle.dump(qtree, f)
    f.close()


    #qtree.plot()
    #plt.show()
    """