import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Wedge
import matplotlib.cm as cm

from time import sleep

def create_graph(name,num_hunters,target_distance,distance_accuracy):
    plt.ion()
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    ax.set_title(name,fontsize=30)
    lines = []
    colors = cm.rainbow(np.linspace(0, 1, num_hunters))
    for hunter_num in range(num_hunters):
        line, = ax.plot(0,0,'-',color = colors[hunter_num],alpha = 0.2)
        point, = ax.plot(0,0,'*',color = colors[hunter_num])
        center = [0,0]
        r = target_distance + distance_accuracy
        circle = Wedge(center, r, 0, 360, width=2*distance_accuracy,alpha=0.3,color = colors[hunter_num])
        patch, = ax.add_patch(circle),
        lines += [line,point,patch],
    group_line, = ax.plot(0,0,'-',color = 'b')
    group_point, = ax.plot(0,0,'*',color = 'b',label = 'Group center')
    ax.legend()
    graph = group_line,group_point,lines,fig, ax
    return graph

def update_graph(graph,hunter_trajectory,group_tragectory):
    group_line,group_point,lines,fig, ax = graph
    for i,(line,point,patch) in enumerate(lines):
        trajectoty = hunter_trajectory[i]
        center = trajectoty[-1]
        line.set_data(*trajectoty.T)
        point.set_data(*center)
        patch.set_center(center)
        
    group_line.set_data(*group_tragectory.T)
    group_point.set_data(*group_tragectory[-1])
    
    x = hunter_trajectory[:,:,0]
    y = hunter_trajectory[:,:,1]
    ax.set_xlim(x.min(),x.max())
    ax.set_ylim(y.min(),y.max())
    fig.canvas.draw()

    
def model_hunter_learning(name,policy,env,hunter_start_position = None):
    graph = create_graph(name,env.num_hunters,env.target_distance,env.distance_accuracy)
    if hunter_start_position == 0:
        hunter_start_position = np.zeros((env.num_hunters,2))
    env.reset(hunter_start_position)
    
    state = env.state
    while True:
        policy(torch.Tensor(state))
        action,_ = policy.sample_action()
        env.step(action)
        state = env.state
        update_graph(graph,env.hunter_trajectory,env.group_trajectory)