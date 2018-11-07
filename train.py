import matplotlib.pyplot as plt
import torch, torch.nn as nn
import time 
from IPython import  display
import torch.optim as optim
import numpy as np

def select_action(policy,state):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state_mu,state_logsigma = policy(torch.Tensor(np.hstack(state)))
    action,log_prob = policy.sample_action()
    # Add log probability of our chosen action to our history    
    if policy.policy_history.dim() != 0:
        policy.policy_history = torch.cat([policy.policy_history, log_prob.unsqueeze(0)])
    else:
        policy.policy_history = (log_prob.unsqueeze(0))
    return action

def update_policy(policy,optimizer,gamma):
    R = 0
    rewards = []
    
    # Discount future rewards back to the present using gamma
    for r in (policy.reward_episode[::-1]):
        R = r + gamma * R
        rewards.insert(0,R)
    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std())
    
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, rewards).mul(-1), -1))
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #Save and intialize episode history counters
    policy.loss_history.append(loss.data[0])
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = torch.Tensor()
    policy.reward_episode= []

def train(policy,env,episodes,learning_rate = 0.0001,gamma = 0.9, verbose=True):
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    for episode in range(episodes):
        env.reset() # Reset environment and record the starting state
        state = env.state
        done = False       
    
        for time in range(1000):
            action = select_action(policy,state)
            # Step through environment using chosen action
            hunter_shift = action
            reward = env.step(hunter_shift)
            state = env.state
            # Save reward
            policy.reward_episode.append(reward)
        
        
        
        update_policy(policy,optimizer,gamma)
        
        if verbose and episode % 10 == 0:
            
            display.clear_output(wait=True)

            plt.figure(figsize=(16, 6))
            plt.subplot(221)
            plt.title("reward")
            plt.xlabel("#iteration")
            plt.ylabel("reward")
            plt.plot(policy.reward_history, label = 'reward')
            plt.subplot(222)
            victim = np.array(env.victim_trajectory).T
            plt.plot(victim[0],victim[1], label = 'victim')
            plt.plot(victim[0][0], victim[1][0], 'o', label = 'initiial_victim')
            hunter = np.array(env.hunter_trajectory).T
            plt.plot(hunter[0], hunter[1], label = 'hunter')
            plt.plot(hunter[0][0], hunter[1][0], 'o', label = 'initiial_hunter')
            plt.legend()
           
        
            plt.show()
        
            print('Episode {}\tLast length: {:5d}\tLast reward: {:.2f}'.format(episode, time, policy.reward_history[-1]))

