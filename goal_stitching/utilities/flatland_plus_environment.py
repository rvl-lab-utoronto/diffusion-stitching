import numpy as np
import gymnasium as gym
from tqdm import trange
import matplotlib.pyplot as plt
import os
class FlatLandPlus(gym.Env): 
    """A simple goal-conditioned navigation task in R^N space.

    The agent is a point that starts at the centre of a unit hypercube in R^N space, where
    N is specified by the user, and the goal is the centre of another facet. 

    The number of facets of the hypercube grows linearly with N, so the total number of state-goal pairs (counting
    repeats as seperate, but not allowing for the same face to be a pair) grows quadratically. 

    observation space is the agent's point in R^N. 

    action space is a vector in R^N determining direction. Norm of vector determines magnitude.

    dynamics: point takes a normalized step of configurable size in R^n space in the direction and magnitude
    of the given action. Default is 1/100 (so most episodes will be about 100 steps long). 

    Ball does NOT have momentum or velocity. 
    """
    # terminated gives True if that env. has been successfully completed
    # truncated gives True if that env. has reached timeout limit (all envs will do this at once)
    def __init__(
        self,
        n_dims = 3,
        step_size = 0.01,
        tolerance = 0.01,
        timeout = 200):
        self.n_dims = n_dims
        self.step_size = step_size
        self.tolerance = tolerance
        self.timeout = timeout
        # normal gym stuff
        self.observation_space = gym.spaces.Box(-np.inf,np.inf,shape=(self.n_dims,),dtype=float)
        self.action_space = gym.spaces.Box(-np.inf,np.inf,shape=(n_dims,),dtype=float)
        self.cur_task_id = -1 # just needed for OGBench inference script compatibility 
        self.start_idx = 0 # just some random defaults
        self.goal_idx = 3
       
    def step(self,
             action):
        # normalizes the action direction
        direction = action/np.linalg.norm(action+1e-8) # small eps. term to stop div by 0
        magnitude = np.clip(np.linalg.norm(action+1e-8),a_max=self.step_size,a_min=0)
        self.state = np.clip(self.state + direction * magnitude,a_max=0.5,a_min=-0.5)

        # updates timeout counter
        self.current_step += 1
        # determines terminations
        terminated = np.linalg.norm(self.state-self.goal) <= self.tolerance
        truncated = self.current_step >= self.timeout
        return self.state,0,terminated,truncated,{}
    def reset(self, seed= 0, options =  {},start_idx=None,goal_idx = None):
        # goal setting stuff
        if start_idx is not None:
            self.start_idx = start_idx
        if goal_idx is not None:
            self.goal_idx = goal_idx
        # resets internal state
        
        self.state = np.zeros(shape = self.n_dims)
        # starts agent at correct location
        dim, sign = np.divmod(self.start_idx,2)
        # sanity check
        assert dim < self.n_dims
        self.state[dim] = sign - 0.5

        # handles goal location creation
        self.goal = np.zeros(shape = self.n_dims)
        dim, sign = np.divmod(self.goal_idx,2)
        assert dim < self.n_dims
        self.goal[dim] = sign - 0.5
        
        # resets timeout tracking
        self.current_step = 0
        return self.state, {'goal':self.goal}
    def render():
        # TODO finish if i decide i want to 
        """ way we handle rendering is using matplotlib to get the figure,
        then saving that through a bunch of conversions to a numpy array"""
        fig = plt.figure()
        np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    def set_start_goal_idx(self,start_idx,goal_idx):
        self.start_idx = start_idx
        self.goal_idx = goal_idx

def make_envs_and_datasets_flatland(n_dims,
                                    dataset_dir = 'utilities/flatland_data',
                                    env_only = False,
                                    single_env = False,
                                    num_envs = 32):
    # Make environment/environments
    if single_env:
        envs = FlatLandPlus(n_dims=n_dims)
    else:
        env_make_list = [lambda: FlatLandPlus(n_dims=n_dims)] * num_envs
        envs = gym.vector.AsyncVectorEnv(env_make_list)
    if env_only:
        return envs
    
    # Load dataset
    dataset = np.load(dataset_dir + '/dataset_n'+str(n_dims) +'.npy',allow_pickle=True).item() # TODO update w/right extension
    return envs, dataset, None # last None thing is to ensure compatability w/OGBench's loader
    