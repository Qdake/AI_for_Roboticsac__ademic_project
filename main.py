import pickle
from plot_result import plot_position_record
from plot_result import plot_qtree
import sys
import gym
import matplotlib.pyplot as plt
from projet_MAPelites_curiosite import MAPelites_curiosite
from projet_NS import novelty_search
from projet_SHINE import SHINE
from deap import base
from deap import creator

## creat class Individual
IND_SIZE = 192 #(5,2,2,10)
#create class
creator.create("FitnessMax",base.Fitness,weights=(1.0,))
creator.create("Individual",list,fitness=creator.FitnessMax,pos=list,profondeur=float)
########################
##### parametres de l'isntance
#########################
display= False
k = sys.argv[1]  # chaine de caractere pour distinguer differentes instances
nb_generation = int(sys.argv[2])
size_pop = int(sys.argv[3])
algo_name = sys.argv[4]
pb_crossover = 0.1
pb_mutation = 0.9


##################################
#######  main
##################################
if __name__ == "__main__":


    # init gym
    env = gym.make('FastsimSimpleNavigation-v0')
    # type d'algo dispo
    assert algo_name in ['SHINE','MAPelites','NS']
    # diff cas
    if algo_name == 'SHINE':
        position_record, qtree, nb_gen_found = SHINE(env,nb_generation=nb_generation, size_pop=size_pop,pb_crossover=pb_crossover,pb_mutation=pb_mutation,display=display)
    if algo_name == 'MAPelites':
        position_record, grid, nb_gen_found = MAPelites_curiosite(env,nb_generation=nb_generation, size_pop=size_pop,pb_crossover=pb_crossover,pb_mutation=pb_mutation,display=display)
    if algo_name == 'NS':
        position_record, arc, nb_gen_found = novelty_search(env,nb_generation=nb_generation, size_pop=size_pop,pb_crossover=pb_crossover,pb_mutation=pb_mutation,display=display)
    # fermer gym
    env.close()
    


    print("nb***",nb_gen_found)
    ##########################################
    ##### sauvegarder le resultat dans un pickle 
    ##########################################
    nfolder = 'log/'+algo_name+'/'
    nfile = algo_name + '_' + k +"_gen_" + str(nb_generation) + "_size_" + str(size_pop)
    nimg =  algo_name + '_' + k +"_gen_" + str(nb_generation) + "_size_" + str(size_pop)

    ## pickle
    data = [position_record,nb_gen_found]
    if algo_name == 'SHINE':
        data.append(qtree)
    ## data = [position_record,nb_gen_found] 
    ## ou data = [position_record,nb_gen_found,qtree] if algo_name == 'SHINE' 
    f = open(nfolder+nfile, 'wb')
    pickle.dump(data, f)
    f.close()
    ## plot 
    #positions_record

    pr = pickle.load( open( nfolder+nfile, "rb" ) )
    position_record = pr[0]
    plot_position_record(position_record,nfolder,nimg)  #plot and save
    # qtree
    if algo_name == 'SHINE':
        qtree = pr[2]
        plot_qtree(qtree,nfolder,nimg)  #plot and save

    plt.title(algo_name)
    plt.show()