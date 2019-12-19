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
import random
from simulation import simulation
import sys
import pickle  

def choix_selon_curiosite(grid, curiosity, h, l, size_pop):
    """choisir une population d'individus dans la grille leurs curiosites
        P[individu] = curiosite_de_individu/somme_curiosite_de_tous_les_individu  
    """
    cases_list = []
    for i in range(h):
        for j in range(l):
            cases_list.append((i,j))
    grid_list = np.resize(grid,(h*l))
    curiosity_list = np.resize(curiosity,(h*l))
    somme = sum(curiosity_list)
    if somme != 0:
        distribution = [c/somme for c in curiosity_list]  # distribution selon curiosite
    else:
        somme = sum([1 if grid_list[i]!= None else 0 for i in range(len(grid_list))]) # compter nombre d'individu dans la grille
        distribution = [1/somme if grid_list[i] != None else 0 for i in range(len(grid_list))] # distribution uniforme
    indices = np.random.choice(list(range(len(grid_list))),size_pop,replace = True,p=distribution)
    return [[cases_list[i],] for i in indices],[grid_list[i] for i in indices]

def MAPelites_curiosite(env,size_pop=50,pb_crossover=0.1, pb_mutation=0.3, nb_generation=100, display=False):

    IND_SIZE = 192
    h_grid = 60
    l_grid = 60
    random.seed()

    #create class
    creator.create("FitnessMax",base.Fitness,weights=(1.0,))
    creator.create("Individual",list,fitness=creator.FitnessMax,pos=list,curiosity=int)
    # toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.normal)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list,  toolbox.individual)
#    toolbox.register("select", tools.selTournament, tournsize=5)

    # initialisation grid
    grid = np.array([[None for i in range(h_grid)] for j in range(l_grid) ])  # hard maze est de taille 600*600
    curiosity = np.array([[0 for i in range(h_grid)] for j in range(l_grid) ])
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
            curiosity[int(ind.bd[0]/10)][int(ind.bd[1]/10)] = 10

    # main boucle
    for gen in range(nb_generation):
        print("generation ",gen)

        # Select the next generation individuals
        parents_pos_in_grid,pop = choix_selon_curiosite(grid,curiosity,h_grid,l_grid, size_pop)    # population est l'ensemble des individus qui presentent dans le grid
        # Clone the selected individuals
        pop = list(map(toolbox.clone, pop))  

        # crossover
        indices = list(range(len(pop)))
        for i,j in zip(indices[::2], indices[1::2]):
            if np.random.random()<pb_crossover:
                #toolbox.mate(pop[i],pop[j])
                tools.cxBlend(pop[i],pop[j],alpha=0.1)
                parents_pos_in_grid[i].append(parents_pos_in_grid[j][0])
                parents_pos_in_grid[j].append(parents_pos_in_grid[i][0])

        #mutation
        for mutant in pop:
            if np.random.random() < pb_mutation:
                tools.mutGaussian(mutant, mu=0.0, sigma=1, indpb=0.1)

        # simulation and MAJ de grid et curiosity
        for i in range(len(pop)):
            pop[i].bd,but_atteint = simulation(env,pop[i],display=display)
            position_record.append(pop[i].bd)
            # si le but est atteint
            """
            if but_atteint:
                print("***********************************************************************")
                print("***************************but atteint MAPelites_curiosite *************************")
                print(gen)
                print("***********************************************************************")
                return position_record,grid,gen 
            """

            # MAJ de la curiosite
            if grid[int(pop[i].bd[0]/10)][int(pop[i].bd[1]/10)] == None:
                # si atteint une case vide
                grid[int(pop[i].bd[0]/10)][int(pop[i].bd[1]/10)] = pop[i]
                for parent_pos in parents_pos_in_grid[i]:
                    curiosity[parent_pos[0]][parent_pos[1]] = 10   # bonus si enfant atteint une position non exploree auparavant
            else:
                # sinon
                for parent_pos in parents_pos_in_grid[i]:
                    curiosity[parent_pos[0]][parent_pos[1]] = max(0,curiosity[parent_pos[0]][parent_pos[1]]-1)
    return position_record,grid,None

