import matplotlib.pyplot as plt
import torch, torch.nn as nn
import time 
from IPython import  display
import torch.optim as optim
import numpy as np
import os
import shutil



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

def compute_value_function(policy,gamma):
    
    R = 0
    rewards = []
    
    # Discount future rewards back to the present using gamma
    for r in (policy.reward_episode[::-1]):
        R = r + gamma * R
        rewards.insert(0,R)

    rewards = torch.FloatTensor(rewards)
    
    return rewards
    

def compute_loss(policy, rewards, baseline_rewards=None):
    
    policy.reward_history.append(np.sum(policy.reward_episode))
    # Scale rewards
    if baseline_rewards is None:
        rewards = (rewards - rewards.mean()) / (rewards.std())
    else:
        rewards = rewards - baseline_rewards
        rewards = (rewards - rewards.mean()) / (rewards.std())
    
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, rewards).mul(-1), -1))
    
    policy.loss_history.append(loss.data[0])

   
    return loss

def compute_baseline_loss(baseline, rewards, baseline_rewards):
    
    # Calculate loss
#     rewards = torch.FloatTensor(rewards)
#     baseline_rewards = torch.cat(baseline_rewards, dim=-1)
    loss = (rewards-baseline_rewards).pow(2).mean()
    
    baseline.loss_history.append(loss.data[0])
    baseline.reward_history.append(torch.cat(baseline.reward_episode, dim=-1).mean().data)
   
    return loss

def play_episode(policy, env, baseline):
    
    for time in range(1000):
        action = select_action(policy, env.state)
        
        #baseline update
        if baseline is not None:
            baseline_reward = baseline(torch.Tensor(np.hstack(env.state)))
            baseline.reward_episode.append(baseline_reward)
        # Step through environment using chosen action    
        # conmpute reward, renew state
        reward = env.step(action) 
        # Save reward
        policy.reward_episode.append(reward)
        
    
def visualize(env, policy, baseline):
    
    plt.figure(figsize=(16, 10))

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

    if baseline is not None:
        plt.subplot(223)
        plt.title("baseline_reward")
        plt.xlabel("#iteration")
        plt.ylabel("reward")
        plt.plot(baseline.reward_history, label = 'reward')

        plt.subplot(224)
        plt.title("baseline_loss")
        plt.xlabel("#iteration")
        plt.ylabel("loss")
        plt.plot(baseline.loss_history, label = 'loss')

    plt.show()
    
def train(policy,env, baseline, episodes,learning_rate = 1e-4,gamma = 0.9, verbose=True, save_policy = True, batch=1):
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    if baseline is not None: b_optimizer = optim.Adam(baseline.parameters(), lr=1e-3)
    
    dirpath = 'train_models'
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.mkdir(dirpath)
    
    loss = 0
    for episode in range(0, episodes):
        env.reset() # Reset environment and record the starting state   
        play_episode(policy, env, baseline)        
        
        values = compute_value_function(policy, gamma)
        
        if baseline is not None:
            # baseline backprop
            baseline_values = torch.cat(baseline.reward_episode, dim=-1)
            b_loss = compute_baseline_loss(baseline, values, baseline_values) 
            b_optimizer.zero_grad()
            b_loss.backward(retain_graph=True)
            b_optimizer.step()
        else:
            baseline_values = None
        
#         if baseline is not None:
#             baseline_rewards = compute_reward(baseline, gamma)
#         else: 
#             baseline_rewards = None
        
        # policy backprop
        loss += compute_loss(policy, values, baseline_values)
        policy.reset_game()    
        if baseline is not None: baseline.reset_game()
            
        #update hunter policy 
        if episode % batch == 0:
            loss /= batch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss=0

        if save_policy and episode % 100 == 0:
            torch.save(policy,dirpath+'/policy_'+str(episode)+'.p') 
        
        if verbose and episode % 10 == 0:
            
            display.clear_output(wait=True)
            visualize(env, policy, baseline)
            
            print('Episode {} \tLast reward: {:.2f}'.format(episode, policy.reward_history[-1]))
            
            if baseline is not None:
                print('Last reward: {:.2f}'.format(baseline.reward_history[-1]))
                