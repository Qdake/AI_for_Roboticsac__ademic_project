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
from simulation import simulation
import sys
import pickle

def es(env,size_pop=50,pb_crossover=0.6, pb_mutation=0.3, nb_generation=100, display=False, verbose=False):

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
    #statistics
    halloffame = tools.HallOfFame(1)
    statistics = tools.Statistics(lambda ind: ind.fitness.values)
    statistics.register("avg", numpy.mean)
    statistics.register("std", numpy.std)
    statistics.register("min", numpy.min)
    statistics.register("max", numpy.max)
        #log
    logbook = tools.Logbook()
    logbook.header = ["gen","nevals"]+ statistics.fields

    # pour plot heatmap
    position_record = []

    # generer la population initiale
    pop = toolbox.population(size_pop)
    # simulation

    for ind in pop:
        ind.bd = simulation(env,ind,display=display)
        position_record.append(ind.bd)
 
    #print(pop[0])
    #print([ind.bd for ind in pop])

    # MAJ archive
    arc = updateNovelty(pop,pop,None)    

    # MAJ fitness
    for ind in pop:
        ind.fitness.values = (ind.novelty,)
   

    # Update the hall of fame with the generated individuals
    halloffame.update(pop)
    record = statistics.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)
    if verbose:
        print(logbook.stream)
    
    offspring = pop
    for gen in range(1, nb_generation+1):
        print("generation ",gen)

        # Select the next generation individuals
        pop = toolbox.select(offspring, size_pop)
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
                tools.mutGaussian(mutant, mu=0.0, sigma=0.000001, indpb=0.1)
                del mutant.fitness.values

        # simulation
        invalid_inds = [ind for ind in pop if ind.fitness.valid == False]
        for ind in invalid_inds:
            ind.bd = simulation(env,ind,display=display)
            position_record.append(ind.bd)
        # MAJ archive
        arc = updateNovelty(pop,pop,arc,k=15)  #Update the novelty criterion (including archive update) 
        # MAJ fitness
        for ind in pop:
            ind.fitness.values = (ind.novelty,)

        # remplacement
        offspring[:] = offspring + pop

            # Update the hall of fame with the generated individuals
        halloffame.update(offspring)
        record = statistics.compile(offspring)
        logbook.record(gen=gen, nevals=len(offspring), **record)
        if verbose:
            print(logbook.stream)

        #if but_atteint:
        #    break
            
    return offspring,logbook, halloffame,position_record






if __name__ == "__main__":

    st = time.time()

    display= False
    env = gym.make('FastsimSimpleNavigation-v0')

    nb_generation = int(sys.argv[2])
    size_pop = int(sys.argv[3])

    but_atteint = False
    #simulation(env,None,True)
    #_,_,_,position_record = es(env,nb_generation=10, size_pop=100,pb_crossover=0.1,pb_mutation=0.9,display=display,verbose=True)
    _,_,_,position_record = es(env,nb_generation=nb_generation, size_pop=size_pop,pb_crossover=0.1,pb_mutation=0.9,display=display,verbose=True)
    env.close()



    from plot_result import plot

    #=================== Traitement du resultat ==========================================================
    k = sys.argv[1]
    nfolder ='log_NS/' 
    nfile = 'position_record_' + k +"_nbgen_"+sys.argv[2]+"_sizepop_"+sys.argv[3]
    nimg = 'position_record_' + k +"_nbgen_"+sys.argv[2]+"_sizepop_"+sys.argv[3]
    plot(position_record,nfolder,nimg,None)  #plot and save

    f = open(nfolder+nfile, 'wb')
    pickle.dump(position_record, f)
    f.close()

    print(but_atteint)
    print(time.time()-st)