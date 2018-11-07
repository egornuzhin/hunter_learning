import numpy as np

class HunterEnvironment:
    def __init__(self,victim_policy,target_distance = 2,distance_accuracy = 1, victim_position=0, hunter_position = [0,0]):
        self.observation_space = 6
        self.action_space = 2
        self.distance_accuracy = distance_accuracy
        self.target_distance = target_distance
        self.victim_policy = victim_policy
        
        # if None - get random position on elipse
        if victim_position is None:
            self.time = np.random.randint(1000)
        else:
            self.time = victim_position
            
        self.victim_position = np.array(self.victim_policy(self.time ))
        self.time += 1
        
        self.hunter_position = np.array(hunter_position)
        self.initial_victim_position = self.victim_position
        self.initial_hunter_position = self.hunter_position
        self.hunter_shift = np.array([0, 0])
        self.hunter_trajectory = []
        self.victim_trajectory = []
        self.distance = np.linalg.norm(self.victim_position-self.hunter_position)
        

    def step(self, hunter_shift):
        hunter_shift = np.array(hunter_shift)
        self.update_positions(hunter_shift)
        self.update_state()
        reward = self.get_reward()
        return reward
    
    def get_reward(self):
        if self.is_hunter_on_target_distance:
            reward = 1
        elif self.is_hunter_closer:
            reward = 0
        else:
            reward = -1
        return reward
        
        
    def update_positions(self,hunter_shift):
        #hunter
        self.hunter_shift = hunter_shift
        self.hunter_position = self.hunter_position+hunter_shift
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
        self.state = [is_hunter_closer,is_hunter_on_target_distance,
                      normalized_victim_shift,self.hunter_shift]
        
        
    def reset(self):
        
        self.__init__(victim_policy=self.victim_policy, target_distance=self.target_distance, 
                      distance_accuracy=self.distance_accuracy, victim_position=None,
                      hunter_position=self.initial_hunter_position)
        
        flip = np.random.uniform()
        if flip<1/2:
            v = np.random.uniform(-1, 1, size=2)
            v = v/np.linalg.norm(v)
            self.hunter_position = self.victim_position + v*self.target_distance
            self.state = [False,True,np.array([0.,0.]),np.array([0.,0.])]
        else:
            self.state = [False,False,np.array([0.,0.]),np.array([0.,0.])]
            v = np.random.uniform(-1, 1, size=2)
            v_norm = np.linalg.norm(v)
            v_new = v/v_norm*(v_norm+1)*self.target_distance*2
            self.hunter_position = self.victim_position +v_norm

        self.hunter_trajectory = [self.hunter_position]
        self.victim_trajectory = [self.victim_position]