import gym
import numpy as np
import time
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
import os, sys
from helpers import *
print("Working dir:"+os.getcwd())
print("Python version:"+sys.version)

import random


# Setup the parameters for the specific problem (you can change all of these if you want to) 
problem_id = 0        # problem_id \in [0:7] generates 8 diffrent problems on which you can train/fine-tune your agent 
reward_hole = -0.01     # should be less than or equal to 0.0 (you can fine tune this  depending on you RL agent choice)
is_stochastic = True  # should be False for A-star (deterministic search) and True for the RL agent

learning_rate = 0.1
discount_rate = 0.8

exploration_rate = 1
er_min = 0.01
er_max = 1

max_episodes = 10000   # you can decide you rerun the problem many times thus generating many episodes... you can learn from them all!
max_iter_per_episode = 2000 # you decide how many iterations/actions can be executed per episode

# Generate the specific problem 
env = LochLomondEnv(problem_id=problem_id, is_stochastic=is_stochastic, reward_hole=reward_hole)

# Let's visualize the problem/env
print(env.desc)

# Create a representation of the state space for use with AIMA A-star
#state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)

q_table = np.zeros([env.observation_space.n, env.action_space.n])

er_curve = np.geomspace(er_max, er_min, max_episodes)
print(er_curve)

# Reset the random generator to a known state (for reproducability)
np.random.seed(12)

####
for ep in range(max_episodes): # iterate over episodes
    observation = env.reset() # reset the state of the env to the starting state
    done = False

    ep_reward = 0
    
    for iter in range(max_iter_per_episode):
        #env.render() # for debugging/develeopment you may want to visualize the individual steps by uncommenting this line

        explore_rand = random.random()
        if explore_rand < exploration_rate:
            action = env.action_space.sample() # your agent goes here (the current agent takes random actions)
        else:
            action = np.argmax(q_table[observation])
        
        new_observation, reward, done, info = env.step(action) # observe what happends when you take the action

        prev_val = q_table[observation, action]
        new_val = np.max(q_table[new_observation])

        q_table[observation, action] = (1 - learning_rate) * prev_val + learning_rate * (reward + discount_rate * new_val)
        observation = new_observation

        # TODO: You'll need to add code here to collect the rewards for plotting/reporting in a suitable manner

        #print("ep,iter,reward,done =" + str(ep) + " " + str(iter)+ " " + str(reward)+ " " + str(done))

        if (done):
            if (reward == reward_hole):
                print(ep,"No")
                #print("We have reached a hole :-( [we can't move so stop trying; just give up]")
                break
            elif (reward == +1.0):
                print(ep,"Success")
                #print("We have reached the goal :-) [stop trying to move; we can't]. That's ok we have achieved the goal]")
                break

        if (iter == max_iter_per_episode-1):
            print(ep,"Nope")
            #print("Ran out of iterations :-( [we failed]")
            break

        exploration_rate = er_curve[ep]
        #exploration_rate = er_min + (er_max - er_min) * np.exp(-0.01*ep)