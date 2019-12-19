#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym, gym_fastsim
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
from novelty_search import NovArchive
from novelty_search import updateNovelty
import random
from simulation import simulation
import sys
import pickle

def novelty_search(env,size_pop=250,pb_crossover=0.1, pb_mutation=0.9, nb_generation=1000, display=False):

    IND_SIZE = 192
    random.seed()

    #create class
    creator.create("FitnessMax",base.Fitness,weights=(1.0,))
    creator.create("Individual",list,fitness=creator.FitnessMax,bd=list,novelty=float)
    # toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.normal)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list,  toolbox.individual)
    # pour plot heatmap
    position_record = []

    # generer la population initiale
    pop = toolbox.population(size_pop)
    # simulation

    for ind in pop:
        ind.bd,but_atteint = simulation(env,ind,display=display)
        position_record.append(ind.bd)
 
    #print(pop[0])
    #print([ind.bd for ind in pop])

    # MAJ archive
    arc = updateNovelty(pop,pop,None)    

    # MAJ fitness
    for ind in pop:
        ind.fitness.values = (ind.novelty,)
    
    offspring = pop
    for gen in range(1, nb_generation+1):
        print("generation ",gen)

        # Select the next generation individuals
        pop = tools.selTournament(offspring, size_pop,5)
        # Clone the selected individuals
        pop = list(map(toolbox.clone, pop))  

        # crossover
        for child1, child2 in zip(pop[::2], pop[1::2]):
            if np.random.random()<pb_crossover:
                #toolbox.mate(child1,child2)
                tools.cxBlend(child1,child2,alpha=0.1)
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
            position_record.append(ind.bd)
            if but_atteint:
                print("***********************************************************************")
                print("***************************but atteint NS *************************")
                print(gen)
                print("***********************************************************************")
                return position_record,arc, gen
        # MAJ archive
        arc = updateNovelty(pop,pop,arc,k=15)  #Update the novelty criterion (including archive update) 
        # MAJ fitness
        for ind in pop:
            ind.fitness.values = (ind.novelty,)          
        # remplacement
        offspring[:] = offspring + pop
            
    return position_record,arc,None