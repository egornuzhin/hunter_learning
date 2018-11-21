import numpy as np
from scipy.spatial.distance import pdist, squareform



class VelocityHunterEnvironment:
    def __init__(self,victim_policy,target_distance = 2,distance_accuracy = 1, time = 0, hunter_position = [0,0], max_acceleration = np.inf):
        self.observation_space = 6
        self.action_space = 2
        self.max_acceleration = max_acceleration
        self.distance_accuracy = distance_accuracy
        self.target_distance = target_distance
        self.victim_policy = victim_policy
        self.time = time
        
        
        self.victim_position = np.array(self.victim_policy(self.time))
        last_victim_position = np.array(self.victim_policy(self.time-1))
        self.victim_shift = self.victim_position - last_victim_position
        self.time += 1
        
        self.hunter_position = np.array(hunter_position)
        self.hunter_shift = np.array([0, 0])
        
        
        self.hunter_trajectory = [self.hunter_position]
        self.victim_trajectory = [self.victim_position]
        
        self.distance = np.linalg.norm(last_victim_position-self.hunter_position)
        self.update_state()
        
        

    def step(self, action):
        self.update_hunter_shift(action)
        self.update_positions()
        self.update_state()
        reward = self.get_reward()
        return reward
    

    def get_c(self,last_shift,desired_shift):
        if np.linalg.norm(last_shift-desired_shift)<= self.max_acceleration:
            c = 1
        else:
            b = last_shift@desired_shift
            a = (desired_shift**2).sum()
            D = b**2-a*((last_shift**2).sum()-self.max_acceleration**2)
            c = max((b-np.sqrt(D))/a,(b+np.sqrt(D))/a)

        return c
        
    def update_hunter_shift(self,action):
        last_shift = self.hunter_shift
        desired_shift = np.array(action)
        c = self.get_c(last_shift,desired_shift)
        if 0<=c and c<=1:
            self.hunter_shift = desired_shift*c
        else:
            k = 1-self.max_acceleration/np.linalg.norm(last_shift)
            self.hunter_shift = last_shift*max(k,0)
            
        
    
    def get_reward(self):
        if self.is_hunter_on_target_distance:
            reward = 1
        elif self.is_hunter_closer:
            reward = 0
        else:
            reward = -1
        return reward
        
        
    def update_positions(self):
        #hunter
        self.hunter_position = self.hunter_position+self.hunter_shift
        self.hunter_trajectory.append(self.hunter_position)
        # victim
        new_victim_position = self.victim_policy(self.time)
        self.victim_shift = new_victim_position-self.victim_position
        self.victim_position = new_victim_position
        self.victim_trajectory.append(self.victim_position)
        
        self.time +=1
    
    def update_state(self):
        new_distance = np.linalg.norm(self.victim_position-self.hunter_position)
        is_hunter_closer = new_distance<self.distance
        is_hunter_on_target_distance = abs(new_distance-self.target_distance)<self.distance_accuracy
        self.distance = new_distance
        normalized_victim_shift = self.victim_shift/np.linalg.norm(self.victim_shift)
        
        
        self.is_hunter_closer = is_hunter_closer
        self.is_hunter_on_target_distance = is_hunter_on_target_distance
        self.state = np.hstack([is_hunter_closer,is_hunter_on_target_distance,
                      normalized_victim_shift,self.hunter_shift])
        
        
    def reset(self):
        
        
        median = self.target_distance
        r = np.random.exponential(median/np.log(2))
        direction = np.random.rand(2)
        radial_shift = r*(direction/np.linalg.norm(direction))
        
        time = np.random.randint(1000)
        victim_position = np.array(self.victim_policy(time))
        hunter_position = victim_position+radial_shift
        
        
        self.__init__(victim_policy=self.victim_policy, target_distance=self.target_distance, 
                      distance_accuracy=self.distance_accuracy, time=time,
                      hunter_position=hunter_position, max_acceleration = self.max_acceleration)
        
        

class ForceHunterEnvironment:
    def __init__(self,victim_policy,target_distance = 2,distance_accuracy = 1, time = 0, hunter_position = [0,0],mass = 1):
        self.observation_space = 8
        self.action_space = 2
        self.distance_accuracy = distance_accuracy
        self.target_distance = target_distance
        self.victim_policy = victim_policy
        self.time = time
        self.mass = mass
        
        self.victim_position = np.array(self.victim_policy(self.time))
        last_victim_position = np.array(self.victim_policy(self.time-1))
        self.victim_shift = self.victim_position - last_victim_position
        self.time += 1
        
        self.hunter_position = np.array(hunter_position)
        self.hunter_shift = np.array([0, 0])
        self.hunter_force = np.array([0, 0])
        
        
        self.hunter_trajectory = [self.hunter_position]
        self.victim_trajectory = [self.victim_position]
        
        self.distance = np.linalg.norm(last_victim_position-self.hunter_position)
        self.update_state()
        
        

    def step(self, action):
        self.update_hunter_shift(action)
        self.update_positions()
        self.update_state()
        reward = self.get_reward()
        return reward
    
    def update_hunter_shift(self,action):
        
        self.hunter_force = np.array(action)
        last_shift = self.hunter_shift
        self.hunter_shift = last_shift+self.hunter_force/self.mass
        
    
    def get_reward(self):
        if self.is_hunter_on_target_distance:
            reward = 1
        elif self.is_hunter_closer:
            reward = 0
        else:
            reward = -1
        return reward
        
        
    def update_positions(self):
        #hunter
        self.hunter_position = self.hunter_position+self.hunter_shift
        self.hunter_trajectory.append(self.hunter_position)
        # victim
        new_victim_position = self.victim_policy(self.time)
        self.victim_shift = new_victim_position-self.victim_position
        self.victim_position = new_victim_position
        self.victim_trajectory.append(self.victim_position)
        
        self.time +=1
    
    def update_state(self):
        new_distance = np.linalg.norm(self.victim_position-self.hunter_position)
        is_hunter_closer = new_distance<self.distance
        is_hunter_on_target_distance = abs(new_distance-self.target_distance)<self.distance_accuracy
        self.distance = new_distance
        normalized_victim_shift = self.victim_shift/np.linalg.norm(self.victim_shift)
        if np.linalg.norm(self.hunter_shift) == 0:
            hunter_shift = np.random.rand(2)
            normalized_hunter_shift = hunter_shift/np.linalg.norm(hunter_shift)
        else:
            normalized_hunter_shift = self.hunter_shift/np.linalg.norm(self.hunter_shift)
        
        
        self.is_hunter_closer = is_hunter_closer
        self.is_hunter_on_target_distance = is_hunter_on_target_distance
        self.state = np.hstack([is_hunter_closer,is_hunter_on_target_distance,
                      normalized_victim_shift,normalized_hunter_shift,self.hunter_force])
        
        
    def reset(self):
        
        
        median = self.target_distance
        r = np.random.exponential(median/np.log(2))
        direction = np.random.rand(2)
        radial_shift = r*(direction/np.linalg.norm(direction))
        
        time = np.random.randint(1000)
        victim_position = np.array(self.victim_policy(time))
        hunter_position = victim_position+radial_shift
        
        
        self.__init__(victim_policy=self.victim_policy, target_distance=self.target_distance, 
                      distance_accuracy=self.distance_accuracy, time=time,
                      hunter_position=hunter_position, mass = self.mass)
        
        
class GroupedHunterEnvironment:
    def __init__(self,
                 target_distance = 2,
                 distance_accuracy = 1,
                 mass = 1,num_hunters = 10, 
                 initial_hunter_positions = None):
        self.observation_space = 11
        self.action_space = 2
        self.distance_accuracy = distance_accuracy
        self.target_distance = target_distance
        self.mass = mass
        self.num_hunters = num_hunters
        self.initial_hunter_positions = initial_hunter_positions
        
        if initial_hunter_positions is None:
             self.hunter_positions = self.get_random_positions()
        else:
            self.hunter_positions = initial_hunter_positions
        
        self.closet_hunter_indices = self.get_closest_hunter_indices()
        
        self.hunter_shift = np.zeros((self.num_hunters,2))
        self.group_shift = np.zeros(2)
        self.hunter_force = np.zeros((self.num_hunters,2))
        
        
        self.hunter_trajectory = np.array(self.hunter_positions)[:,np.newaxis,:]
        
        self.group_position = self.hunter_positions.mean(axis = 0)
        self.group_trajectory = np.array([self.group_position])
        
        self.closed_distances = np.linalg.norm(self.hunter_positions - 
                                               self.hunter_positions[self.closet_hunter_indices],
                                               axis = 1)
        self.distances_to_center = np.linalg.norm(self.hunter_positions - self.group_position,axis = 1)
        self.update_state()
        
    def get_random_positions(self):
        scattering = (self.target_distance + self.distance_accuracy)*self.num_hunters/4
        return np.random.randn(self.num_hunters,2)*scattering
        
    def get_closest_hunter_indices(self):
        closet_hunter_indices = np.argmax(squareform(1/pdist(self.hunter_positions)),axis = 0)
        return closet_hunter_indices

    def step(self, action):
        self.hunter_force = np.array(action)
        self.update_shifts()
        self.update_positions()
        self.update_state()
        reward = self.get_reward()
        return reward
    
    def update_shifts(self):
        last_shift = self.hunter_shift
        self.hunter_shift = last_shift+self.hunter_force/self.mass
        self.group_shift = self.hunter_shift.mean(axis = 0)

    # def get_reward(self):
    #     reward = np.clip(self.is_hunter_at_target_distance.astype(np.int)*2+
    #                      self.is_center_closer.astype(np.int),0,2)-1
    #     return reward

    def get_reward(self):
        reward = self.is_hunter_at_target_distance.astype(np.int)*2+ \
                         self.is_center_closer.astype(np.int)-self.is_hunter_very_close*(-3)
        return reward



    def update_positions(self):
        #hunter
        self.hunter_positions = self.hunter_positions+self.hunter_shift
        self.hunter_trajectory = np.concatenate((self.hunter_trajectory,
                                                   self.hunter_positions[:,np.newaxis,:]),axis = 1)
        # group
        self.group_position = self.hunter_positions.mean(axis = 0)
        self.closet_hunter_indices = self.get_closest_hunter_indices()
        self.group_trajectory = np.append(self.group_trajectory,[self.group_position],axis=0)
    
    def update_state(self):
        new_closed_distances = np.linalg.norm(self.hunter_positions-
                                              self.hunter_positions[self.closet_hunter_indices],
                                              axis = 1)
        is_closest_hunter_closer = new_closed_distances<self.closed_distances
        is_hunter_at_target_distance = np.abs(new_closed_distances-
                                              self.target_distance)<self.distance_accuracy
        #new
        self.is_hunter_very_close = new_closed_distances < self.target_distance-self.distance_accuracy

        self.closed_distances = new_closed_distances
        
        new_distances_to_center = np.linalg.norm(self.hunter_positions-self.group_position,axis = 1)
        is_center_closer = new_distances_to_center<self.distances_to_center
        self.distances_to_center = new_distances_to_center
        
        shift_is_zero = (self.hunter_shift == 0) * (np.flip(self.hunter_shift,1) == 0)
        hunter_shift = np.where(shift_is_zero,np.random.rand(self.num_hunters,2),self.hunter_shift)
        normalized_hunter_shift = hunter_shift/(np.linalg.norm(hunter_shift,axis = 1)[:,np.newaxis])
        normalized_closed_hunter_shift = normalized_hunter_shift[self.closet_hunter_indices]
        
        if np.linalg.norm(self.group_shift) == 0:
            group_shift = np.random.rand(2)
        else:
            group_shift = self.group_shift
        normalized_group_shift = (group_shift/np.linalg.norm(group_shift))
        
        self.is_center_closer = is_center_closer
        self.is_hunter_at_target_distance = is_hunter_at_target_distance
        
        self.state = np.concatenate(
            (is_closest_hunter_closer[:,np.newaxis],
             is_hunter_at_target_distance[:,np.newaxis],
             is_center_closer[:,np.newaxis],
             normalized_hunter_shift,
             normalized_closed_hunter_shift,
             np.tile(normalized_group_shift,(10,1)),
             self.hunter_force),axis = -1)

    def reset(self,initial_hunter_positions=None):
        if initial_hunter_positions is None:
            initial_hunter_positions = self.initial_hunter_positions
        self.__init__(target_distance = self.target_distance,
                      distance_accuracy = self.distance_accuracy,
                      mass = self.distance_accuracy,
                      num_hunters = self.num_hunters,
                      initial_hunter_positions = initial_hunter_positions)
