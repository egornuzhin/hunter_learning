import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Wedge
import matplotlib.cm as cm

from time import sleep

def create_graph(name,num_hunters,target_distance,distance_accuracy):
    plt.ion()
    fig = plt.figure(figsize = (10,20))
    ax_model = fig.add_subplot(211)
    ax_reward = fig.add_subplot(212)
    ax_model.set_title(name,fontsize=30)
    ax_reward.set_title('Reword',fontsize=30)
    lines = []
    colors = cm.rainbow(np.linspace(0, 1, num_hunters))
    for hunter_num in range(num_hunters):
        line, = ax_model.plot(0,0,'-',color = colors[hunter_num],alpha = 0.2)
        point, = ax_model.plot(0,0,'*',color = colors[hunter_num])
        center = [0,0]
        r = target_distance + distance_accuracy
        circle = Wedge(center, r, 0, 360, width=2*distance_accuracy,alpha=0.3,color = colors[hunter_num])
        patch, = ax_model.add_patch(circle),
        lines += [line,point,patch],
    group_line, = ax_model.plot(0,0,'-',color = 'b')
    group_point, = ax_model.plot(0,0,'*',color = 'b',label = 'Group center')
    reward_line, = ax_reward.plot(0, 0)
    ax_model.legend()
    graph = group_line,group_point,lines,reward_line,fig, (ax_model,ax_reward)
    return graph

def update_graph(graph,hunter_trajectory,group_tragectory,rewards):
    group_line,group_point,lines,reward_line,fig, (ax_model,ax_reward) = graph
    for i,(line,point,patch) in enumerate(lines):
        trajectoty = hunter_trajectory[i]
        center = trajectoty[-1]
        line.set_data(*trajectoty.T)
        point.set_data(*center)
        patch.set_center(center)
        
    group_line.set_data(*group_tragectory.T)
    group_point.set_data(*group_tragectory[-1])
    
#     reward_line.set_data(np.arange(len(ema_rewards)),ema_rewards)
    reward_line.set_data(np.arange(len(rewards)),rewards)
    ax_reward.set_xlim(np.arange(len(rewards)).min(),np.arange(len(rewards)).max())
    ax_reward.set_ylim(rewards.min()-1,rewards.max()+1)
    
    x = hunter_trajectory[:,:,0]
    y = hunter_trajectory[:,:,1]
    if x.max()-x.min()>y.max()-y.min():
        dy = ((x.max()-x.min())-(y.max()-y.min()))/2
        ax_model.set_xlim(x.min(),x.max())
        ax_model.set_ylim(y.min()-dy,y.max()+dy)
    else:
        dx = ((y.max()-y.min())-(x.max()-x.min()))/2
        ax_model.set_xlim(x.min()-dx,x.max()+dx)
        ax_model.set_ylim(y.min(),y.max())
    fig.canvas.draw()

    
def model_hunter_learning(name,policy,env,hunter_start_position = None):
    graph = create_graph(name,env.num_hunters,env.target_distance,env.distance_accuracy)
    if hunter_start_position == 0:
        hunter_start_position = np.zeros((env.num_hunters,2))
    env.reset(hunter_start_position)
    rewards = np.array([])
    state = env.state
    while True:
        policy(torch.Tensor(state))
        action,_ = policy.sample_action()
        reward = env.step(action).sum()
        rewards = np.append(rewards,reward)
        state = env.state
        update_graph(graph,env.hunter_trajectory,env.group_trajectory,rewards)