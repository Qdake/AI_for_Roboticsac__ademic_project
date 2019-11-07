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
from novelty_search import NovArchive
from novelty_search import updateNovelty

def simulation(env,genotype,display=True):
    global size_nn
    nn=SimpleNeuralControllerNumpy(5,2,2,5)
    if genotype != None:
        nn.set_parameters(genotype)
    observation = env.reset()
    if(display):
        env.enable_display()
    then = time.time()

    for i in range(2000):
        env.render()
        action=nn.predict(observation)
        action = [i * env.maxVel for i in action]
        observation,reward,done,info=env.step(action)
        print("Step %d Obs=%s  reward=%f  dist. to objective=%f  robot position=%s  End of ep=%s" % (i, str(observation), reward, info["dist_obj"], str(info["robot_pos"]), str(done)))
        if(display):
            time.sleep(0.01)
        if done:
            break

    now = time.time()

    print("%d timesteps took %f seconds" % (i, now - then))

    #print(env.get_robot_pos())    
    return env.get_robot_pos()    # x,y,theta    ?? pourquoi theta??? to do


def es(env,size_pop=50,pb_crossover=0.6, pb_mutation=0.3, nb_generation=100, display=False, verbose=True):

    IND_SIZE = 72
    
    #create class
    creator.create("FitnessMax",base.Fitness,weights=(1.0,))
    creator.create("Individual",list,fitness=creator.FitnessMax,pos=list)
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

    # generer la population initiale
    pop = toolbox.population(size_pop)
    # evaluation
    for ind in pop:
        ind.fitness.values = eval_nn(env,ind)

    # Update the hall of fame with the generated individuals
    halloffame.update(pop)
    record = statistics.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)
    if verbose:
        print(logbook.stream)

    
    for gen in range(1, nb_generation+1):
        print("generation ",gen)

        # Select the next generation individuals
        offspring = toolbox.select(pop, size_pop)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))  

        # crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random()<pb_crossover:
                toolbox.mate(child1,child2)
                del child1.fitness.values
                del child2.fitness.values


        #mutation
        for mutant in offspring:
            if np.random.random() < pb_mutation:
                tools.mutGaussian(mutant, mu=0.0, sigma=1, indpb=0.1)
                del mutant.fitness.values

        # evaluation
        invalid_inds = [ind for ind in offspring if ind.fitness.valid == False]
        for ind in invalid_inds:
            ind.fitness.values = eval_nn(env,ind)

        # remplacement
        pop[:] = offspring

            # Update the hall of fame with the generated individuals
        halloffame.update(pop)
        record = statistics.compile(pop)
        logbook.record(gen=gen, nevals=len(pop), **record)
        if verbose:
            print(logbook.stream)
            
    return pop,logbook, halloffame








nn=SimpleNeuralControllerNumpy(5,2,2,5)
print(len(nn.get_parameters()))


display= True
env = gym.make('FastsimSimpleNavigation-v0')

#simulation(env,None,True)
env.close()