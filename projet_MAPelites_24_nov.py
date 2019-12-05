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
        #print("Step %d Obs=%s  reward=%f  dist. to objective=%f  robot position=%s  End of ep=%s" % (i, str(observation), reward, info["dist_obj"], str(info["robot_pos"]), str(done)))
        if(display):
            print("sleep,sleep")
            time.sleep(0.01)
        if (info["dist_obj"]<=env.goalRadius):
            but_atteint = True
            break


    now = time.time()

    #print("%d timesteps took %f seconds" % (i, now - then))

    x,y,theta = env.get_robot_pos()    # x,y,theta    ?? pourquoi theta??? to do
    return [int(x),int(y)]    


def es(env,size_pop=50,pb_crossover=0.6, pb_mutation=0.3, nb_generation=100, display=False, verbose=False):

    IND_SIZE = 72
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

    # initialisation grid
    grid = [[None for i in range(600)] for j in range(600) ]  # hard maze est de taille 600*600
#    for i in range(600):
#        for j in range(600):
#            if grid[i][j] != None:
#                print(i," ",j)
#    print("fin print1")
 
    # pour plot heatmap
    position_record = []

    # generer la population initiale
    pop = toolbox.population(size_pop)
    print("****2   len(pop) **",len(pop)) #debug
    # simulation

    for ind in pop:
        ind.bd = simulation(env,ind,display=display)
        position_record.append(ind.bd)
        if grid[ind.bd[0]][ind.bd[1]] == None:
            grid[ind.bd[0]][ind.bd[1]] = ind
    print("*******2.1 len(pop)******",len(pop))
    for i in range(600):
        for j in range(600):
            if grid[i][j] != None:
                print(i," ",j)
    print("fin print2")

    #print(pop[0])
    #print([ind.bd for ind in pop])

    # MAJ archive
    pop = [grid[i][j] for i in range(600) for j in range(600) if (grid[i][j] != None)]
    print("*******3 len(pop)******",len(pop)) #debug
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
            ind.bd = simulation(env,ind,display=display)
            position_record.append(ind.bd)
            if grid[ind.bd[0]][ind.bd[1]] == None:
                grid[ind.bd[0]][ind.bd[1]] = ind

        # maj archive            
        pop = [grid[i][j] for i in range(600) for j in range(600) if not(grid[i][j] == None)]
        arc = updateNovelty(pop,pop,None)    #Update the novelty criterion (including archive update) 

        # MAJ fitness
        for ind in pop:
            ind.fitness.values = (ind.novelty,)

        # pas de remplacement
        # remplacement
        # offspring[:] = offspring + pop

            # Update the hall of fame with the generated individuals
        halloffame.update(pop)
        record = statistics.compile(pop)
        logbook.record(gen=gen, nevals=len(pop), **record)
        if verbose:
            print(logbook.stream)

        #if but_atteint:
        #    break
            
    return pop,logbook, halloffame,position_record






if __name__ == "__main__":

    st = time.time()

    nn=SimpleNeuralControllerNumpy(5,2,2,5)
    #print(len(nn.get_parameters()))


    display= False
    env = gym.make('FastsimSimpleNavigation-v0')

    but_atteint = False
    #simulation(env,None,True)
    #_,_,_,position_record = es(env,nb_generation=10, size_pop=100,pb_crossover=0.1,pb_mutation=0.9,display=display,verbose=True)
    _,_,_,position_record = es(env,nb_generation=1000, size_pop=250,pb_crossover=0.1,pb_mutation=0.9,display=display,verbose=True)
    env.close()



    from plot_result import plot

    #=================== Traitement du resultat ==========================================================
    k = sys.argv[1]
    nfile = 'log_MAPelites_24_nov/'
    nimg = 'position_record_24_nov_2130' + str(k)
    plot(position_record,nfile,nimg)  #plot and save

    print(but_atteint)
    print(time.time()-st)

