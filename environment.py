import numpy as np

class HunterEnvironment:
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
    
#     def update_hunter_shift(self,action):
#         force = action.data.numpy()
#         last_shift = self.hunter_shift
#         self.hunter_shift = last_shift+force/self.mass

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
            k = 1-self.max_acceleration/max(np.linalg.norm(last_shift), 1e-10)
            self.hunter_shift = last_shift*max(k,0)
            
#             print('k', k)
#         print('c',c)
        
    
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
        self.state = [is_hunter_closer,is_hunter_on_target_distance,
                      normalized_victim_shift,self.hunter_shift]
        
        
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