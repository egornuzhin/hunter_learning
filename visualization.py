import matplotlib.pyplot as plt
import numpy as np
import torch
from time import sleep
from matplotlib.patches import Wedge

def create_graph(name,target_distance,distance_accuracy):
    plt.ion()
    fig = plt.figure(figsize = (10,20))
    ax_model = fig.add_subplot(211)
    ax_reward = fig.add_subplot(212)
    ax_model.set_title(name,fontsize=30)
    ax_reward.set_title('Reword',fontsize=30)
    hunter_line, = ax_model.plot(0, 0, 'r-',label = 'Hunter trajectory',alpha=0.5)
    hunter_position_point, = ax_model.plot(0, 0, 'r*',label = 'Hunter position')
    victim_line, = ax_model.plot(0, 0, 'g-',label = 'Victim trajectory',alpha=0.5)
    victim_position_point, = ax_model.plot(0, 0, 'g*',label = 'Victim position')
    r = target_distance + distance_accuracy
    circle = Wedge([0,0], r, 0, 360, width=2*distance_accuracy,alpha=0.3,color = 'g')
    victim_patch = ax_model.add_patch(circle)
    reward_line, = ax_reward.plot(0, 0)

    ax_model.legend()
    graph = (hunter_line,hunter_position_point,victim_line,victim_position_point,victim_patch,reward_line),fig, (ax_model, ax_reward)
    return graph

def update_graph(graph,hunter_trajectoty,victim_trajectoty,rewards):
    (hunter_line,hunter_position_point,victim_line,victim_position_point,victim_patch,reward_line),fig, (ax_model, ax_reward) = graph
    x = np.r_[hunter_trajectoty[0],victim_trajectoty[0]]
    y = np.r_[hunter_trajectoty[1],victim_trajectoty[1]]
    max_range = max(x.max()-x.min(),y.max()-y.min())
    if x.max()-x.min()>y.max()-y.min():
        dy = ((x.max()-x.min())-(y.max()-y.min()))/2
        ax_model.set_xlim(x.min(),x.max())
        ax_model.set_ylim(y.min()-dy,y.max()+dy)
    else:
        dx = ((y.max()-y.min())-(x.max()-x.min()))/2
        ax_model.set_xlim(x.min()-dx,x.max()+dx)
        ax_model.set_ylim(y.min(),y.max())
    hunter_line.set_data(*hunter_trajectoty)
    hunter_position_point.set_data(*hunter_trajectoty.T[-1])
    victim_line.set_data(*victim_trajectoty)
    victim_position_point.set_data(*victim_trajectoty.T[-1])
    center = victim_trajectoty.T[-1]
    victim_patch.set_center(center)
    reward_line.set_data(np.arange(len(rewards)),rewards)
    ax_reward.set_xlim(np.arange(len(rewards)).min(),np.arange(len(rewards)).max())
    ax_reward.set_ylim(rewards.min()-1,rewards.max()+1)
    fig.canvas.draw()

    
def model_hunter_learning(name,policy,env,hunter_start_position = [0,0]):
    graph = create_graph(name,env.target_distance,env.distance_accuracy)
    env.reset()
    env.hunter_position = hunter_start_position
    env.hunter_trajectory = [hunter_start_position]
    state = env.state
    rewards = np.array([])
    while True:
        policy(torch.Tensor(state))
        action,_ = policy.sample_action()
        reward = env.step(action)
        rewards = np.append(rewards,reward)
        state = env.state
        update_graph(graph,np.array(env.hunter_trajectory).T,np.array(env.victim_trajectory).T,rewards)
            
        



