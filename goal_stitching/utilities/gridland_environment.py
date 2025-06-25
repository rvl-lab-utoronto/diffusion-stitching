import numpy as np
import gymnasium as gym
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
class GridLand(gym.Env): 
    """A simple goal-conditioned navigation task in a grid in a 2D grid.

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
        n_size = 3,
        step_size = 0.01,
        tolerance = 0.05,
        timeout = 500,
        burn_time = 10):
        self.n_size = n_size
        self.step_size = step_size
        self.tolerance = tolerance
        self.timeout = timeout
        # normal gym stuff
        self.observation_space = gym.spaces.Box(-np.inf,np.inf,shape=(2,),dtype=float)
        self.action_space = gym.spaces.Box(-np.inf,np.inf,shape=(2,),dtype=float)
        self.cur_task_id = -1 # just needed for OGBench inference script compatibility 
        self.start_idx = 'T0' # just some random defaults
        self.goal_idx = 'B' + str(n_size)
        
        ### env. hyperparameters not meant to be varied
        self.block_percent = 0.6
        self.hallway_percent = 0.4
        self.difficulty = 'easy'
        assert self.block_percent + self.hallway_percent == 1.0
        ### Proper environment set up ###
        # calculates key point coordinate spaces - should be n+1 of each.
        keypoint_offset = (self.hallway_percent/2.0)/self.n_size
        box_width = 2*(self.block_percent)/self.n_size
        # note - both of following keypoints are for both x and y axes
        self.keypoints = np.linspace(start = -1.0 + keypoint_offset, stop = 1.0 - keypoint_offset, num = self.n_size+1)
        
        # calculates box coordinates
        self.box_bottoms = np.linspace(start = -1.0 + keypoint_offset*2, stop = 1.0 - keypoint_offset*2 - box_width, num = self.n_size,endpoint=True)
        self.box_tops = self.box_bottoms + box_width

        # stores stuff
        self.box_width = box_width

        # hard difficulty stuff
        self.burn_time = burn_time # how long you can spend in the lava before BURNING!!!
       
    def step(self,
             action):
        """ if you disagree with how this method was written
        a] don't care
        b] didn't ask
        c] YOU try writing this efficently
        d] even though it's cancer asymptotically, this environment never gets large enough for it to matter
        e] plus spinus minus linus plus ratio
        f] it's code redundant but minimizes branching checks. I think
        """
        # normalizes the action direction
        direction = action/np.linalg.norm(action+1e-8) # small eps. term to stop div by 0
        magnitude = np.clip(np.linalg.norm(action+1e-8),a_max=self.step_size,a_min=0)

        # handles new state and running into blocks n stuff
        # I know this is quadratic in length, but this will never get that big so I don't really care that much 
        new_state = self.state + direction * magnitude
        # easy case - if point tries to move into block, "push it out" 
        if self.difficulty == 'easy':
            for i in range(len(self.box_bottoms)):
                for j in range(len(self.box_bottoms)):
                    # checks for collision 
                    if self.box_bottoms[i] < new_state[0] < self.box_tops[i] and self.box_bottoms[j] < new_state[1] < self.box_tops[j]:
                        # case where x displacement less than y displacement
                        if (min(abs(self.box_bottoms[i] - new_state[0]), abs(new_state[0] - self.box_tops[i])) < # how deep into block point is, x 
                            min(abs(self.box_bottoms[j] - new_state[1]), abs(new_state[1] - self.box_tops[j]))): # how deep into block point is, y
                            # checks if closer to left or right of block
                            if abs(self.box_bottoms[i] - new_state[0]) < abs(new_state[0] - self.box_tops[i]):
                                new_state[0] = self.box_bottoms[i] # shoves to bottom
                            else:
                                new_state[0] = self.box_tops[i] # shoves to top
                        # case where y displacement less than x displacement
                        else:
                            # checks if closer to top or bottom of block
                            if abs(self.box_bottoms[j] - new_state[1]) < abs(new_state[1] - self.box_tops[j]):
                                new_state[1] = self.box_bottoms[j] # shoves to left
                            else:
                                new_state[1] = self.box_tops[j] # shoves to right
        # hard case - let point move into block, make it possible to "burn" and fail episode
        elif self.difficulty == 'hard':
            collided = False
            for i in range(len(self.box_bottoms)):
                for j in range(len(self.box_bottoms)):
                    # checks for collision
                    if self.box_bottoms[i] < new_state[0] < self.box_tops[i] and self.box_bottoms[j] < new_state[1] < self.box_tops[j]:
                        collided = True
                        self.burn_counter += 1
                        if self.burn_counter >= self.burn_time:
                            self.burnt = True
            # no collision - resets burn counter.
            if not collided:
                self.burn_counter = 0
            # checks to see if already fully burnt
            if self.burnt:
                new_state = self.state # sets new state to old state (cannot make progress anymore)


        self.state = np.clip(new_state,a_min=-1.0,a_max=1.0)

        # updates timeout counter
        self.current_step += 1
        # determines terminations
        terminated = np.linalg.norm(self.state-self.goal) <= self.tolerance
        truncated = self.current_step >= self.timeout
        return self.state,int(terminated),terminated,truncated,{} # state, reward, terminated, truncated, info 
    def reset(self, seed= 0, options =  {},start_idx=None,goal_idx = None,difficulty = 'hard'):
        # goal setting stuff
        if start_idx is not None:
            self.start_idx = start_idx
        if goal_idx is not None:
            self.goal_idx = goal_idx
        if options is not None: # not sure why this is needed but it throws a fit w/o it
            if 'start_idx' in options:
                self.start_idx = options['start_idx']
            if 'goal_idx' in options:
                self.goal_idx = options['goal_idx']
        # difficulty stuff
        assert difficulty == 'easy' or difficulty == 'hard','ERROR: Invalid Gridland Difficulty!'
        self.difficulty = difficulty
        # resets internal state. Note - for now assumes start_idx and goal_idx are in the same form (string or tuple)
        if type(self.start_idx) is str:
            # case where it is defined as T0, B5, etc.
            self.start = np.array(self.convert_pos_str_to_coords(self.start_idx))
            self.state = np.array(self.convert_pos_str_to_coords(self.start_idx))
            self.goal = np.array(self.convert_pos_str_to_coords(self.goal_idx))
        else:
            # case where user gives X-Y coordinate tuple
            self.start = np.array((self.keypoints[self.start_idx[0]],self.keypoints[self.start_idx[1]]))
            self.state = np.array((self.keypoints[self.start_idx[0]],self.keypoints[self.start_idx[1]]))
            self.goal = np.array((self.keypoints[self.goal_idx[0]],self.keypoints[self.goal_idx[1]]))
        self.current_step = 0
        self.burn_counter = 0 # how many consecutive steps have been in lava
        self.burnt = False # if we've burnt this episode (you stay there until timeout)
        return self.state, {'goal':self.goal,'keypoints':self.keypoints}
    def render():
        # TODO finish if i decide i want to 
        """ way we handle rendering is using matplotlib to get the figure,
        then saving that through a bunch of conversions to a numpy array"""
        fig = plt.figure()
        np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    def set_start_goal_idx(self,start_idx,goal_idx):
        self.start_idx = start_idx
        self.goal_idx = goal_idx

    def convert_pos_str_to_coords(self,pos_str):
        """
        pretty simple helper method
        
        Note - index always goes left to right/top to bottom
        """
        side = pos_str[0]
        index = int(pos_str[1:])
        if side == 'T':
            return (self.keypoints[index], self.keypoints[-1])
        elif side == 'B':
            return (self.keypoints[index], self.keypoints[0])
        elif side == 'L':
            return (self.keypoints[0], self.keypoints[index])
        elif side == 'R':
            return (self.keypoints[-1], self.keypoints[index])
        else:
            assert False, 'ERROR: Invalid Position String!'
    def get_mpl_plot(self,show_keypoints = False):
        block_colour = 'grey'
        if self.difficulty == 'hard':
            #block_colour = 'red'
            #block_colour = '#FF964F'
            block_colour = '#d86666'

        fig, ax = plt.subplots(figsize=(6, 6))
        if show_keypoints:
            for x in self.keypoints:
                for y in self.keypoints:
                    ax.scatter(x,y)
        for x in self.box_bottoms:
            for y in self.box_bottoms:
                square = patches.Rectangle((x,y),
                                            self.box_width, 
                                            self.box_width, 
                                            edgecolor=block_colour, 
                                            facecolor=block_colour,
                                            zorder=-1)
                ax.add_patch(square)
        ax.scatter(self.start[0],self.start[1],label='Start',zorder=100000000000,marker='p',color='#8DDF76',s=3000/self.n_size,edgecolors='black')
        ax.scatter(self.goal[0],self.goal[1],label='Goal',zorder=100000000000000,marker='*',color = '#FFBF00',s=3250/self.n_size,edgecolors='black')

        square = patches.Rectangle((-1.05,-1.05), 2.1, 2.1, edgecolor='black', facecolor='none',lw=2.5)
        ax.add_patch(square)
        ax.axis('off')
        ax.set_title('Gridland')
        ax.axis("tight")  # gets rid of white border
        #fig.legend()
        return fig,ax

def make_envs_and_datasets_gridland(n_size,
                                    dataset_dir = 'utilities/gridland_data_new',
                                    env_only = False,
                                    single_env = False,
                                    goal_cond = False,
                                    num_envs = 32,
                                    override_dir = False):
    # Make environment/environments
    if single_env:
        envs = GridLand(n_size=n_size)
    else:
        env_make_list = [lambda: GridLand(n_size=n_size)] * num_envs
        envs = gym.vector.AsyncVectorEnv(env_make_list)
    if env_only:
        return envs
    
    # Load dataset
    if override_dir:
        dataset = np.load(dataset_dir,allow_pickle=True).item()
    else:
        if not goal_cond:
            dataset = np.load(dataset_dir + '/dataset_n'+str(n_size) +'.npy',allow_pickle=True).item() # TODO update w/right extension
        if goal_cond:
            dataset = np.load(dataset_dir + '/dataset_n'+str(n_size) +'_gc'+'.npy',allow_pickle=True).item() # have no clue what that todo is about
    
    return envs, dataset, None # last None thing is to ensure compatability w/OGBench's loader

def find_trajectory_keypoints(trajectory, env, tolerance = 0.06):
    """ Tolerance during collection was 0.025
    """
    keypoints = env.keypoints
    trajectory_annealed = []
    trajectory_double_annealed = []
    x_indices_annealed = []
    x_indices = []
    y_indices = []
    # x pass
    for state in trajectory:
        x = state[0]
        for i,keypoint in enumerate(keypoints):
            if np.abs(x-keypoint) < tolerance:
                trajectory_annealed.append(state)
                x_indices_annealed.append(i)

    # y pass
    for s_i, state in enumerate(trajectory_annealed):
        y = state[1]
        for j,keypoint in enumerate(keypoints):
            if np.abs(y-keypoint) < tolerance:
                trajectory_double_annealed.append(state)
                y_indices.append(j)
                x_indices.append(x_indices_annealed[s_i])
    
    assert len(y_indices) == len(x_indices), 'Error!'
    visited_intersections = set()
    visited_intersections_indexes = set()
    for index in range(len(x_indices)):
        visited_intersections.add((keypoints[x_indices[index]],keypoints[y_indices[index]]))
        visited_intersections_indexes.add((x_indices[index],y_indices[index]))
    return np.array(list(visited_intersections)), np.array(list(visited_intersections_indexes))

def get_trajectory_sets(trajectories):
    # each trajectory needs to be a set of keypoint pairs (themselves sets)
    trajectories_set = set()
    for trajectory in trajectories:
        # each trajectory is a (node x 2) shape array
        trajectory_set = set()
        for point in trajectory:
            trajectory_set.add(tuple([point[0],point[1]]))
        trajectory_set = frozenset(trajectory_set)
        trajectories_set.add(trajectory_set)
    return trajectories_set

def filter_successful_trajectories(trajectory_set,goals):
    # TODO generalize this to the multigoal context
    # for now "goals" only accepts a single goal 
    filtered_set = set()
    for trajectory in trajectory_set:
        if goals in trajectory:
            filtered_set.add(trajectory)
    return filtered_set