import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal


class HunterPolicy(nn.Module):
    def __init__(self,env,max_action = 1):
        super(HunterPolicy, self).__init__()
        self.max_action = max_action
        self.state_space = env.observation_space
        self.action_space = env.action_space


        
        self.l1 = nn.Linear(self.state_space, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, self.action_space*2)
                
        # Episode policy and reward history 
        self.policy_history = torch.Tensor()#.cuda()
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        
    def get_action(self,latent_action):
        action = self.max_action*latent_action*(torch.exp(-(latent_action**2).sum(dim = -1))/torch.norm(latent_action,dim = -1)).unsqueeze(-1)
        return action
    
    def get_log_jacobian(self,latent_action):
        return 2*((latent_action**2).sum()-np.log(self.max_action))-np.log(2)
        
    def log_prob(self,latent_action):
        latent_log_prob = self.latent_action_dist.log_prob(latent_action).sum(dim = -1)
        log_jacobian = self.get_log_jacobian(latent_action)
        log_prob = latent_log_prob+log_jacobian
        return log_prob
    
    def sample_action(self):
        latent_action = self.latent_action_dist.sample()
        action = self.get_action(latent_action)
        return action, self.log_prob(latent_action)
        
    def forward(self, state):    
            model = torch.nn.Sequential(
                self.l1,
                nn.ELU(),
                self.l2,
                nn.ELU(),
                self.l3
            )
            output = model(state)
            mu,logsigma = torch.chunk(output,2,dim = -1)
            self.latent_action_dist = Normal(mu,(2*logsigma).exp())
            return mu,logsigma
        
class VictimPolicy:

    def eight_victim_policy(t):
        phi = (2*np.pi)/30*t*0.1/2
        r = 10*2

        x = r * np.sin(phi)
        y = r * np.sin(phi)*np.cos(phi)
        return np.array([x,y])

    def circle_victim_policy(t):
        phi = (2*np.pi)/30*t*0.1
        r = 10

        x = r * np.sin(phi)
        y = r * np.cos(phi)
        return np.array([x,y])

    def ellipse_victim_policy(t):
        phi = (2*np.pi)/30*t*0.1
        r = 10

        x = 2*r * np.sin(phi)
        y = r * np.cos(phi)
        return np.array([x,y])

    def triangle_victim_policy(t):
        scale = 15
        speed = 0.01
        phi = (t*speed)%3
        a_vec = np.array([1/2,3**(1/2)/2])
        b_vec = np.array([1/2,-3**(1/2)/2])
        c_vec = np.array([-1, 0])
        out = a_vec*min(1,phi)+b_vec*max(min(1,phi-1),0)+c_vec*max(min(1,phi-2),0)
        return out*scale


    def spiral_victim_policy(t):
        phi = (2*np.pi)/30*t*0.1
        r = 10+2*np.sin(phi*2)**2

        x = 2*r * np.sin(phi)+(t/20)
        y = r * np.cos(phi)+(t/10)
        return np.array([x,y])