#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym, gym_fastsim
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
import time
import numpy as np
from deap import base
from deap import creator
from deap import tools
import numpy 
import matplotlib.pyplot as plt
from novelty_search import NovArchive
from novelty_search import updateNovelty
import random
from simulation import simulation
import sys
import pickle

def MAPelites_ns(env,size_pop=50,pb_crossover=0.1, pb_mutation=0.9, nb_generation=250, display=False):

    IND_SIZE = 192
    random.seed()

    #create class
    creator.create("FitnessMax",base.Fitness,weights=(1.0,))
    creator.create("Individual",list,fitness=creator.FitnessMax,pos=list,novelty=float)
    # toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.normal)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list,  toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate",tools.cxBlend,alpha=0.1)

    # initialisation grid
    grid = np.array([[None for i in range(60)] for j in range(60) ])
    # pour plot heatmap
    position_record = []

    # generer la population initiale
    pop = toolbox.population(size_pop)
    # simulation

    for ind in pop:
        ind.bd,but_atteint = simulation(env,ind,display=display)
        position_record.append(ind.bd)
        if grid[int(ind.bd[0]/10)][int(ind.bd[1]/10)] == None:
            grid[int(ind.bd[0]/10)][int(ind.bd[1]/10)] = ind

    # MAJ archive
    pop = [grid[i][j] for i in range(60) for j in range(60) if (grid[i][j] != None)]
    arc = updateNovelty(pop,pop,None)    


    # MAJ fitness
    for ind in pop:
        ind.fitness.values = (ind.novelty,)

    for gen in range(1, nb_generation+1):
        print("generation ",gen)

        # Select the next generation individuals
        pop = toolbox.select(pop, size_pop)    # population est l'ensemble des individus qui presentent dans le grid
        # Clone the selected individuals
        pop = list(map(toolbox.clone, pop))  

        # crossover
        for child1, child2 in zip(pop[::2], pop[1::2]):
            if np.random.random()<pb_crossover:
                toolbox.mate(child1,child2)
                del child1.fitness.values
                del child2.fitness.values

        #mutation
        for mutant in pop:
            if np.random.random() < pb_mutation:
                tools.mutGaussian(mutant, mu=0.0, sigma=1, indpb=0.1)
                del mutant.fitness.values

        # simulation
        invalid_inds = [ind for ind in pop if ind.fitness.valid == False]
        for ind in invalid_inds:
            ind.bd,but_atteint = simulation(env,ind,display=display)
            # si le but est atteint
            if but_atteint:
                return position_record,grid,gen 
            position_record.append(ind.bd)
            if grid[int(ind.bd[0]/10)][int(ind.bd[1]/10)] == None:
                grid[int(ind.bd[0]/10)][int(ind.bd[1]/10)] = ind

        # maj archive            
        pop = [grid[i][j] for i in range(60) for j in range(60) if not(grid[i][j] == None)]
        arc = updateNovelty(pop,pop,None)    #Update the novelty criterion (including archive update) 

        # MAJ fitness
        for ind in pop:
            ind.fitness.values = (ind.novelty,)

            
    return position_record,grid,None




