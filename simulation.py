import gym, gym_fastsim
from fixed_structure_nn_numpy import SimpleNeuralControllerNumpy
def simulation(env,genotype,display=True):
    global but_atteint
    global size_nn
    nn=SimpleNeuralControllerNumpy(5,2,2,5)#72
    nn=SimpleNeuralControllerNumpy(5,2,2,10)#192
    if genotype != None:
        nn.set_parameters(genotype)
    observation = env.reset()
    if(display):
        env.enable_display()
#    then = time.time()

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
            return [0,0]


#    now = time.time()

    #print("%d timesteps took %f seconds" % (i, now - then))

    x,y,theta = env.get_robot_pos()    # x,y,theta    ?? pourquoi theta??? to do
    return [int(x),int(y)]    
