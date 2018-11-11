import matplotlib.pyplot as plt
import numpy as np
import torch
from time import sleep

def create_graph(name):
    plt.ion()
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    ax.set_title(name,fontsize=30)
    hunter_line, = ax.plot(0, 0, 'r-',label = 'Hunter trajectory')
    hunter_position_point, = ax.plot(0, 0, 'r*',label = 'Hunter position')
    victim_line, = ax.plot(0, 0, 'g-',label = 'Victim trajectory')
    victim_position_point, = ax.plot(0, 0, 'g*',label = 'Victim position')
    
    ax.legend()
    graph = (hunter_line,hunter_position_point,victim_line,victim_position_point),fig, ax
    return graph

def update_graph(graph,hunter_trajectoty,victim_trajectoty):
    (hunter_line,hunter_position_point,victim_line,victim_position_point),fig, ax = graph
    x = np.r_[hunter_trajectoty[0],victim_trajectoty[0]]
    y = np.r_[hunter_trajectoty[1],victim_trajectoty[1]]
    ax.set_xlim(x.min(),x.max())
    ax.set_ylim(y.min(),y.max())
    hunter_line.set_data(*hunter_trajectoty)
    hunter_position_point.set_data(*hunter_trajectoty.T[-1])
    victim_line.set_data(*victim_trajectoty)
    victim_position_point.set_data(*victim_trajectoty.T[-1])
    fig.canvas.draw()

    
def model_hunter_learning(name,policy,env,hunter_start_position = [0,0]):
    graph = create_graph(name)
    env.reset()
    env.hunter_position = hunter_start_position
    env.hunter_trajectory = [hunter_start_position]
    state = env.state
    while True:
        policy(torch.Tensor(np.hstack(state)))
        action,_ = policy.sample_action()
        env.step(action)
        state = env.state
        update_graph(graph,np.array(env.hunter_trajectory).T,np.array(env.victim_trajectory).T)
            
        



