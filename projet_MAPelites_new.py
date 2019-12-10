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

def simulation(env,genotype,display=True):
    global but_atteint
    global size_nn
    nn=SimpleNeuralControllerNumpy(5,2,2,5)
    if genotype != None:
        nn.set_parameters(genotype)
    observation = env.reset()
    if(display):
        env.enable_display()
    then = time.time()

    for i in range(1000):
        env.render()
        action=nn.predict(observation)
        action = [i * env.maxVel for i in action]
        observation,reward,done,info=env.step(action)
        if(display):
            print("sleep,sleep")
            time.sleep(0.01)
        if (info["dist_obj"]<=env.goalRadius):
            but_atteint = True
            break

    now = time.time()

    x,y,theta = env.get_robot_pos()    # x,y,theta    ?? pourquoi theta??? to do
    return [int(x),int(y)]    

def choix_selon_curiosite(grid, curiosity, h, l, size_pop):
    grid_list = np.resize(grid,(h*l));
    curiosity_list = np.resize(curiosity,(h*l));
    distribution = [1/(c+1) for c in curiosity_list]
    somme = sum(distribution)
    distribution = [i/somme for i in distribution]
    indices = np.random.choice(list(range(len(grid_list))),size_pop,replace = True,p=distribution)
    return [grid_list[i] for i in indices]

def es(env,size_pop=50,pb_crossover=0.6, pb_mutation=0.3, nb_generation=100, display=False, verbose=False):

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
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate",tools.cxBlend,alpha=0.1)

    # initialisation grid
    grid = [[None for i in range(h_grid)] for j in range(l_grid) ]  # hard maze est de taille 600*600
    curiosity = [[0 for i in range(h_grid)] for j in range(l_grid) ]
    # pour plot heatmap
    position_record = []

    # generer la population initiale
    pop = toolbox.population(size_pop)

    # simulation
    for ind in pop:
        ind.bd = simulation(env,ind,display=display)
        position_record.append(ind.bd)
        if grid[ind.bd[0]][ind.bd[1]] == None:
            grid[ind.bd[0]][ind.bd[1]] = ind
            curiosity[ind.bd[0]][ind.bd[1]] = 10

    # main boucle
    for gen in range(nb_generation):
        print("generation ",gen)

        # Select the next generation individuals
        pop = choix_selon_curiosite(grid,curiosity,h_grid,l_grid, size_pop)    # population est l'ensemble des individus qui presentent dans le grid

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

        # simulation and MAJ de grid et curiosity
        invalid_inds = [ind for ind in pop if ind.fitness.valid == False]
        for ind in invalid_inds:
            ind.bd = simulation(env,ind,display=display)
            position_record.append(ind.bd)
            if grid[ind.bd[0]][ind.bd[1]] == None:
                grid[ind.bd[0]][ind.bd[1]] = ind
                curiosity[ind.bd[0]][ind.bd[1]] = 10;
            else:
                curiosity[ind.bd[0]][ind.bd[1]] = min(0,curiosity[ind.bd[0]][ind.bd[1]]-1)

    return grid,position_record






if __name__ == "__main__":

    st = time.time()

    display= False
    env = gym.make('FastsimSimpleNavigation-v0')

    nb_generation = int(sys.argv[2])
    size_pop = int(sys.argv[3])

    but_atteint = False
    grid,position_record = es(env,nb_generation=nb_generation, size_pop=size_pop,pb_crossover=0.1,pb_mutation=0.9,display=display,verbose=True)
    env.close()



    from plot_result import plot

    #=================== Traitement du resultat ==========================================================
    k = sys.argv[1]
    nfolder = 'log_MAPelites/'
    nfile = 'position_record_' + k +"_nbgen_"+sys.argv[2]+"_sizepop_"+sys.argv[3]
    nimg = 'position_record_' + k +"_nbgen_"+sys.argv[2]+"_sizepop_"+sys.argv[3]
    plot(position_record,nfolder,nimg,None)  #plot and save

    f = open(nfolder+nfile, 'wb')
    pickle.dump(position_record, f)
    f.close()
    
    print(but_atteint)
    print(time.time()-st)

