import torch as torch
from scipy.spatial.distance import pdist, squareform
from torch.nn import functional as F

class GroupedClusteringHunterEnvironment:
    def __init__(self,
                 comfort_zone_radius = 2,
                 group_center_radius= 1000,
                 mass = 1,num_hunters = 10, 
                 initial_hunter_positions = None, use_cuda = False):
        self.use_cuda = use_cuda
        self.observation_space = 12
        self.action_space = 2
        self.comfort_zone_radius = comfort_zone_radius
        self.group_center_radius = group_center_radius
        self.mass = mass
        self.num_hunters = num_hunters
        self.initial_hunter_positions = initial_hunter_positions
        
        if initial_hunter_positions is None:
             self.hunter_positions = self.get_random_positions()
        else:
            self.hunter_positions = initial_hunter_positions
        self.closet_hunter_indices = self.get_closest_hunter_indices()
        
        self.hunter_shift = torch.zeros((self.num_hunters,2))
        self.group_shift = torch.zeros(2)
        self.hunter_force = torch.zeros((self.num_hunters,2))
        
        if use_cuda:
            self.hunter_shift = self.hunter_shift.cuda()
            self.group_shift = self.group_shift.cuda()
            self.hunter_force = self.hunter_force.cuda()
                
        self.hunter_trajectory = self.hunter_positions.unsqueeze(1)
        
        self.group_position = self.hunter_positions.mean(dim = 0)
        self.group_trajectory = self.group_position.unsqueeze(0)
        
        self.closed_distances = torch.norm(self.hunter_positions - 
                                               self.hunter_positions[self.closet_hunter_indices],
                                               dim = 1)
        self.distances_to_center = torch.norm(self.hunter_positions - self.group_position,dim = 1)
        self.update_state()
        
    def get_random_positions(self):
        scattering = max(self.comfort_zone_radius*self.num_hunters,self.group_center_radius)
        positions = torch.randn(self.num_hunters,2)*scattering
        if self.use_cuda:
            positions = positions.cuda()
        return positions
    
    def get_closest_hunter_indices(self):
        x = self.hunter_positions.transpose(0,1).unsqueeze(0).repeat(self.num_hunters,1,1)
        closet_hunter_indices = torch.argmin(F.pairwise_distance(x,x.transpose(0,2)),dim = 0)
        return closet_hunter_indices
        
        
    def step(self, action):
        if self.use_cuda:
            action = action.cuda()
        self.hunter_force = action
        self.update_shifts()
        self.update_positions()
        self.update_state()
        reward = self.get_reward()
        return reward
    
    def update_shifts(self):
        last_shift = self.hunter_shift
        self.hunter_shift = last_shift+self.hunter_force/self.mass
        self.group_shift = self.hunter_shift.mean(dim = 0)
        
    
    def get_reward(self):
        gc_reward = torch.clamp(self.is_hunter_inside_group_center.type(torch.int)*2+
                            self.is_center_closer.type(torch.int),0,2)-1
        cz_penalty = self.does_closest_hunter_break_cz.type(torch.int)
        return gc_reward - cz_penalty
        
        
    def update_positions(self):
        #hunter
        self.hunter_positions = self.hunter_positions+self.hunter_shift
        self.hunter_trajectory = torch.cat((self.hunter_trajectory,
                                                   self.hunter_positions.unsqueeze(1)),dim = 1)
        # group
        self.group_position = self.hunter_positions.mean(dim = 0)
        self.closet_hunter_indices = self.get_closest_hunter_indices()
        self.group_trajectory = torch.cat([self.group_trajectory,self.group_position.unsqueeze(0)],dim=0)
    
    def update_state(self):
        new_closed_distances = torch.norm(self.hunter_positions-
                                              self.hunter_positions[self.closet_hunter_indices],
                                              dim = 1)
        is_closest_hunter_closer = new_closed_distances<self.closed_distances
        self.closed_distances = new_closed_distances
        does_closest_hunter_break_cz = self.closed_distances<self.comfort_zone_radius
        
        
        new_distances_to_center = torch.norm(self.hunter_positions-self.group_position,dim = 1)
        is_center_closer = new_distances_to_center<self.distances_to_center
        self.distances_to_center = new_distances_to_center
        is_hunter_inside_group_center = self.distances_to_center < self.group_center_radius
        
        
        shift_is_zero_mask = ((self.hunter_shift == 0) * (self.hunter_shift.flip(1) == 0))
        hunter_shift = self.hunter_shift.clone()
        if self.use_cuda:
            rand_shift = torch.randn(shift_is_zero_mask.sum()).cuda()
        else:
            rand_shift = torch.randn(shift_is_zero_mask.sum())
        hunter_shift[shift_is_zero_mask] = rand_shift
        normalized_hunter_shift = hunter_shift/(torch.norm(hunter_shift,dim = 1).unsqueeze(1))
        normalized_closed_hunter_shift = normalized_hunter_shift[self.closet_hunter_indices]
        
        if torch.norm(self.group_shift) == 0:
            group_shift = torch.randn(2)
            if self.use_cuda:
                group_shift = group_shift.cuda()
        else:
            group_shift = self.group_shift
        normalized_group_shift = (group_shift/torch.norm(group_shift))
        
        self.is_closest_hunter_closer = is_closest_hunter_closer
        self.does_closest_hunter_break_cz = does_closest_hunter_break_cz
        self.is_center_closer = is_center_closer
        self.is_hunter_inside_group_center = is_hunter_inside_group_center
        self.state = torch.cat(
            (is_closest_hunter_closer.unsqueeze(1).type(torch.float),
             does_closest_hunter_break_cz.unsqueeze(1).type(torch.float),
             is_center_closer.unsqueeze(1).type(torch.float),
             is_hunter_inside_group_center.unsqueeze(1).type(torch.float),
             normalized_hunter_shift,
             normalized_closed_hunter_shift,
             normalized_group_shift.repeat(self.num_hunters,1),
             self.hunter_force),dim = -1)

        
    def reset(self,initial_hunter_positions = None):
        if initial_hunter_positions is None:
            initial_hunter_positions = self.initial_hunter_positions
        self.__init__(comfort_zone_radius = self.comfort_zone_radius,
                      group_center_radius = self.group_center_radius,
                      mass = self.mass,
                      num_hunters = self.num_hunters,
                      initial_hunter_positions = initial_hunter_positions,
                      use_cuda = self.use_cuda)