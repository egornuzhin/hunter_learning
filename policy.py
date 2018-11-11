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
    
    def reset_game(self):
        
        self.policy_history = torch.Tensor()
        self.reward_episode = []
        
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
        
class Baseline(nn.Module):
    def __init__(self,env):
        super(Baseline, self).__init__()
        
        self.state_space = env.observation_space
        
        self.model = torch.nn.Sequential(
                nn.Linear(self.state_space, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, 1)
            )
                
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
        
    def reset_game(self):
        self.reward_episode = []
              
    def forward(self, state):    

            baseline = self.model(state)

            return baseline
                
