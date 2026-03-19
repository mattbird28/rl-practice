#!/usr/bin/env python
# coding: utf-8

# ## Sarsa equations
# - $Q(s_{t},a_{t}) = Q(s_{t},a_{t}) + \alpha (r_{t+1}+\gamma Q(s_{t+1},a_{t+1})-Q(s_{t},a_{t}))$
# 
# Based on
# https://www.geeksforgeeks.org/sarsa-reinforcement-learning/

# - $\text{learning_rate} = \alpha$
# -  $\text{discount_rate} = \gamma$
# 
# - $\text{exploration_rate} = \epsilon$
# - $\text{max_exploration_rate} =\max{\epsilon}$
# - $\text{min_exploration_rate} = \min{\epsilon}$
# - $\text{exploration_decay_rate} = \epsilon \text{ decay rate}$

# Step 1: Importing the required libraries

# In[14]:


import numpy as np
import gym
import random
import time
from IPython.display import clear_output


# Step 2: Building the environment

# In[20]:


#Building the environment
env = gym.make('FrozenLake-v1')


# Step 3: Initializing different parameters

# In[21]:


#Defining the different parameters
exploration_rate = 0.8
num_episodes = 10000
max_steps_per_episode = 100
learning_rate = 0.85
discount_rate = 0.95

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size,action_space_size))
print(q_table)


# Step 4: Defining utility functions to be used in the learning process

# In[22]:


#Function to choose the next action
def choose_action(state):
    action=0
    if np.random.uniform(0, 1) < exploration_rate:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])
    return action

#Function to learn the Q-value
def update(state, state2, reward, action, action2):
    predict = q_table[state, action]
    target = reward + discount_rate * q_table[state2, action2]
    q_table[state, action] = q_table[state, action] + learning_rate * (target - predict)


# Step 5: Training the learning agent

# In[23]:


#Initializing the reward
rewards_all_episodes = []

# Starting the SARSA learning
for episode in range(num_episodes):
  
    state1 = env.reset()
    action1 = choose_action(state1)
    done = False
    rewards_current_episode = 0
    
    for step in range(max_steps_per_episode):
        
        #Getting the next state
        state2, reward, done, info = env.step(action1)

        #Choosing the next action
        action2 = choose_action(state2)

        #Learning the Q-value
        update(state1, state2, reward, action1, action2)

        state1 = state2
        action1 = action2
        
        #Updating the respective vaLues        
        rewards_current_episode += 1

        #If at the end of learning process
        if done:
            break
            
    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000

print("********Average reward per thousand episodes********\n")
for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

# Print updated Q-table
print("\n\n********Q-table********\n")
print(q_table)


# Step 6: Evaluating the performance

# In[19]:


for episode in range(3):
    # initialize new episode params
    state = env.reset()
    done = False
    print("*****EPISODE ", episode+1, "*****\n\n\n\n")
    time.sleep(1)

    for step in range(max_steps_per_episode):        
        # Show current state of environment on screen
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)
        # Choose action with highest Q-value for current state
        action = np.argmax(q_table[state,:])
        # Take new action
        new_state, reward, done, info = env.step(action)

        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print("****You reached the goal!****")
                time.sleep(3)
            else:
                print("****You fell through a hole!****")
                time.sleep(3)
            clear_output(wait=True)
            break           

        # Set new state
        state = new_state

env.close()


# In[ ]:




