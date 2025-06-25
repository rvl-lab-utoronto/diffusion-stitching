import numpy as np
import math
import gymnasium as gym
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
class Decagon(gym.Env): 
    """ A simple goal-conditioned navigation task in a grid in a 2D grid.

    The agent is a point that starts along the edge of a grid, and must travel to another indexed
    point on the grid.

    How many grid lines there are varies as a user parameter n.

    The number of starting points grows exponentially with n,

    observation space is the agent's point in R^2.

    action space is a vector in R^2 determining direction. Norm of vector determines magnitude of step.

    dynamics: point takes a normalized step of configurable size in R^2 space in the direction and magnitude
    of the given action. Default is 1/100 

    Ball does NOT have momentum or velocity. 
    """
    # terminated gives True if that env. has been successfully completed
    # truncated gives True if that env. has reached timeout limit (all envs will do this at once)
    def __init__(
        self,
        timeout = 500,
        step_size = 0.02,
        tolerance = 0.05,
        ):
        
        self.observation_space = gym.spaces.Box(-np.inf,np.inf,shape=(2,),dtype=float)
        self.action_space = gym.spaces.Box(-np.inf,np.inf,shape=(2,),dtype=float)
        self.cur_task_id = -1 # just needed for OGBench inference script compatibility 

        # sets up points
        self.point_dict = self.make_point_dict() # this is pointname -> coordinate

        # other environment stuff
        self.step_size = step_size
        self.tolerance = tolerance
        self.timeout = timeout
    def step(self,action):
        # normalizes the action direction
        direction = action/np.linalg.norm(action+1e-8) # small eps. term to stop div by 0
        magnitude = np.clip(np.linalg.norm(action+1e-8),a_max=self.step_size,a_min=0)
        new_state = self.state + direction * magnitude
        self.state = np.clip(new_state,a_min=-1.0,a_max=1.0)
        # updates timeout counter
        self.current_step += 1
        # determines terminations
        terminated = False
        truncated = self.current_step >= self.timeout
        return self.state,int(terminated),terminated,truncated,{} # state, reward, terminated, truncated, info 

    def reset(self, seed= 0, options = {}):
        
        self.state = self.point_dict['A'] # eh
        if options is not None: # not sure why this is needed but it throws a fit w/o it
            if 'start_point' in options:
                self.state = self.point_dict[options['start_point']]
            
        # difficulty stuff
        
        # case where user gives X-Y coordinate tuple
        self.current_step = 0
        return self.state, {}

    def get_mpl_plot(self):
        
        fig, ax = plt.subplots(figsize=(6, 6))
        

        for point in self.point_dict:
            point_coords = self.point_dict[point]
            plt.scatter(point_coords[0],point_coords[1],s=250)

        ax.axis('off')
        #ax.set_title('Decagon')
        ax.axis("tight")  # gets rid of white border  
        return fig,ax
    def make_point_dict(self):
        points = {}
        center_x = 0
        center_y = 0
        radius = 1
        point_names = ['A','B','C','D','E','F','G','H','I','J']
        for i in range(10):
            angle = 2 * math.pi * i / 10  # Divide the circle into 10 equal parts
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.update({point_names[i]:np.array((x, y))})
        return points

    def get_trajectory_landmarks(self,trajectory,tolerance = 0.1):
        """
        takes in the trajectory as a list in coordinate space, gives a list of landmarks
        it visits in order
        """
        # storage for the landmarks visited
        landmark_visitations = []
        # gets reverse landmark dict that goes point -> landmark
        reverse_landmark_dict = {tuple(v): k for k, v in self.point_dict.items()}
        for state in trajectory:
            # sees if it is close to any of the landmarks
            for landmark_grid_point in reverse_landmark_dict:
                if np.linalg.norm(landmark_grid_point - state) < tolerance:
                    landmark = reverse_landmark_dict[landmark_grid_point]
                    if landmark not in landmark_visitations:
                        landmark_visitations.append(landmark)
        return np.array(landmark_visitations)
    
def make_envs_decagon(
                    single_env = False,
                    num_envs = 32):
    # Make environment/environments
    if single_env:
        envs = Decagon()
    else:
        env_make_list = [lambda: Decagon()] * num_envs
        envs = gym.vector.AsyncVectorEnv(env_make_list)
    return envs, None, None # last None thing is to ensure compatability w/OGBench's loader


