#!/usr/bin/python
# -*- coding: utf-8 -*-

import gym, gym_fastsim
import time
import numpy as np
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
from novelty_search import NovArchive
from novelty_search import updateNovelty
import random
from arbre_quaternaire import Quadtree
import pickle
import sys
from simulation import simulation 

def choix_a_roulette(population_list, size_pop):
    ''' choix a roulette en utilisant la distribution proportionnelle a la profondeur des individu 
        P(individu) = pow( individu.profondeur, 4) 
    '''
    profondeurs = [ind.profondeur for ind in population_list]
    distribution = [1/pow(4,profondeur) for profondeur in profondeurs]
    somme = sum(distribution)
    distribution = [i/somme for i in distribution]
    indices = np.random.choice(list(range(len(population_list))),size_pop,replace = True,p=distribution)
    return [population_list[i] for i in indices]

def dist(x,y):
    ''' calculer la distance euclidienne entre x et y
    ''' 
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
                tools.cxBlend(child1,child2,alpha=0.1)

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
            # si le but est atteint, le programme s'arrete
            if but_atteint:
                print("***********************************************************************")
                print("***************************but atteint SHINE *************************")
                print(gen)
                print("***********************************************************************")
                return position_record, arbre, gen

    return position_record, arbre, None
