#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 21:18:05 2019

@author: qiu-adm
"""
import gym
import pickle
from plot_result import plot_position_record
from plot_result import plot_qtree
import sys
from projet_MAPelites_curiosite import MAPelites_curiosite
from projet_MAPelites_ns import MAPelites_ns
from projet_NS import novelty_search
from projet_SHINE import SHINE
import multiprocessing
import matplotlib.pyplot as plt
from deap import base
from deap import creator

## creat class Individual
IND_SIZE = 192 #(5,2,2,10)
#create class
creator.create("FitnessMax",base.Fitness,weights=(1.0,))
creator.create("Individual",list,fitness=creator.FitnessMax,pos=list,profondeur=float)

def run(k,nb_generation,size_pop,algo_name):
    ########################
    ##### parametres de l'isntance
    #########################
    display= False
    #k = sys.argv[1]  # chaine de caractere pour distinguer differentes instances
    #nb_generation = int(sys.argv[2])
    #size_pop = int(sys.argv[3])
    #algo_name = sys.argv[4]
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
       
    ## pickle
    data = [position_record,nb_gen_found]
    if algo_name == 'SHINE':
        data.append(qtree)
        
    return data

    
n1 = int(sys.argv[1])
n2 = int(sys.argv[2])
algo_name = sys.argv[3]

multiprocessing.freeze_support()
cpus = multiprocessing.cpu_count()
pool = multiprocessing.Pool(cpus)
results = []
for k in range(n1,n2):
    result = pool.apply_async(run,args=(k,1000,250,algo_name,))
    results.append(result)


pool.close()
pool.join()

results = [result.get() for result in results]
for k in range(n1,n2):
    ##########################################
    ##### sauvegarder le resultat dans un pickle 
    ##########################################
    nfolder = 'log/'+algo_name+'/'
    nfile = algo_name + '_' + str(k) +"_gen_" + sys.argv[2] + "_size_"
    nimg =  algo_name + '_' + str(k) +"_gen_" + sys.argv[2] + "_size_"
    
    ## data = [position_record,nb_gen_found] 
    ## ou data = [position_record,nb_gen_found,qtree] if algo_name == 'SHINE' 
    f = open(nfolder+nfile, 'wb')
    pickle.dump(results[k], f)
    f.close()
    ## plot 
    #positions_record
    plot_position_record(results[k][0],nfolder,nimg)  #plot and save
    # qtree
    if algo_name == 'SHINE':
        plot_qtree(results[k][2],nfolder,nimg)  #plot and save   