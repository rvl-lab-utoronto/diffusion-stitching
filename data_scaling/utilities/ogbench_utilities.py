import dataclasses
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from ogbench import make_env_and_datasets
from tqdm import trange
from collections import defaultdict
import gymnasium
import os
import urllib.request

import gymnasium
import gymnasium as gym # yeah yeah I know double imports sue me 
import numpy as np
from tqdm import tqdm
import numpy as np
import torch
import random
from typing import Optional
import wandb
import uuid

DEFAULT_DATASET_DIR = '~/.ogbench/data'
DATASET_URL = 'https://rail.eecs.berkeley.edu/datasets/ogbench'
def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)

def load_ogbench_data_env(env_name,config):
    """ Why this wrapper? to remove redundant processing code """
    env, train_dataset, _ = make_env_and_datasets(env_name,compact_dataset=False)
    dataset_class = {
            'GCDataset': GCDataset,
            'HGCDataset': HGCDataset,
        }['GCDataset']
    train_dataset = dataset_class(Dataset.create(**train_dataset),config)
    return env, train_dataset
def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)

def load_dataset(dataset_path, ob_dtype=np.float32, action_dtype=np.float32, compact_dataset=False):
    """Load OGBench dataset.

    Args:
        dataset_path: Path to the dataset file.
        ob_dtype: dtype for observations.
        action_dtype: dtype for actions.
        compact_dataset: Whether to return a compact dataset (True, without 'next_observations') or a regular dataset
            (False, with 'next_observations').

    Returns:
        Dictionary containing the dataset. The dictionary contains the following keys: 'observations', 'actions',
        'terminals', and 'next_observations' (if `compact_dataset` is False) or 'valids' (if `compact_dataset` is True).
    """
    file = np.load(dataset_path)

    dataset = dict()
    for k in ['observations', 'actions', 'terminals']:
        if k == 'observations':
            dtype = ob_dtype
        elif k == 'actions':
            dtype = action_dtype
        else:
            dtype = np.float32
        dataset[k] = file[k][...].astype(dtype)

    # Example:
    # Assume each trajectory has length 4, and (s0, a0, s1), (s1, a1, s2), (s2, a2, s3), (s3, a3, s4) are transition
    # tuples. Note that (s4, a4, s0) is *not* a valid transition tuple, and a4 does not have a corresponding next state.
    # At this point, `dataset` loaded from the file has the following structure:
    #                  |<--- traj 1 --->|  |<--- traj 2 --->|  ...
    # -------------------------------------------------------------
    # 'observations': [s0, s1, s2, s3, s4, s0, s1, s2, s3, s4, ...]
    # 'actions'     : [a0, a1, a2, a3, a4, a0, a1, a2, a3, a4, ...]
    # 'terminals'   : [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  1, ...]

    if compact_dataset:
        # Compact dataset: We need to invalidate the last state of each trajectory so that we can safely get
        # `next_observations[t]` by using `observations[t + 1]`.
        # Our goal is to have the following structure:
        #                  |<--- traj 1 --->|  |<--- traj 2 --->|  ...
        # -------------------------------------------------------------
        # 'observations': [s0, s1, s2, s3, s4, s0, s1, s2, s3, s4, ...]
        # 'actions'     : [a0, a1, a2, a3, a4, a0, a1, a2, a3, a4, ...]
        # 'terminals'   : [ 0,  0,  0,  1,  1,  0,  0,  0,  1,  1, ...]
        # 'valids'      : [ 1,  1,  1,  1,  0,  1,  1,  1,  1,  0, ...]

        dataset['valids'] = 1.0 - dataset['terminals']
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = np.minimum(dataset['terminals'] + new_terminals, 1.0).astype(np.float32)
    else:
        # Regular dataset: Generate `next_observations` by shifting `observations`.
        # Our goal is to have the following structure:
        #                       |<- traj 1 ->|  |<- traj 2 ->|  ...
        # ----------------------------------------------------------
        # 'observations'     : [s0, s1, s2, s3, s0, s1, s2, s3, ...]
        # 'actions'          : [a0, a1, a2, a3, a0, a1, a2, a3, ...]
        # 'next_observations': [s1, s2, s3, s4, s1, s2, s3, s4, ...]
        # 'terminals'        : [ 0,  0,  0,  1,  0,  0,  0,  1, ...]

        ob_mask = (1.0 - dataset['terminals']).astype(bool)
        next_ob_mask = np.concatenate([[False], ob_mask[:-1]])
        dataset['next_observations'] = dataset['observations'][next_ob_mask]
        dataset['observations'] = dataset['observations'][ob_mask]
        dataset['actions'] = dataset['actions'][ob_mask]
        new_terminals = np.concatenate([dataset['terminals'][1:], [1.0]])
        dataset['terminals'] = new_terminals[ob_mask].astype(np.float32)

    return dataset


def download_datasets(dataset_names, dataset_dir=DEFAULT_DATASET_DIR):
    """Download OGBench datasets.

    Args:
        dataset_names: List of dataset names to download.
        dataset_dir: Directory to save the datasets.
    """
    # Make dataset directory.
    dataset_dir = os.path.expanduser(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

    # Download datasets.
    dataset_file_names = []
    for dataset_name in dataset_names:
        dataset_file_names.append(f'{dataset_name}.npz')
        dataset_file_names.append(f'{dataset_name}-val.npz')
    for dataset_file_name in dataset_file_names:
        dataset_file_path = os.path.join(dataset_dir, dataset_file_name)
        if not os.path.exists(dataset_file_path):
            dataset_url = f'{DATASET_URL}/{dataset_file_name}'
            print('Downloading dataset from:', dataset_url)
            response = urllib.request.urlopen(dataset_url)
            tmp_dataset_file_path = f'{dataset_file_path}.tmp'
            with tqdm.wrapattr(
                open(tmp_dataset_file_path, 'wb'),
                'write',
                miniters=1,
                desc=dataset_url.split('/')[-1],
                total=getattr(response, 'length', None),
            ) as file:
                for chunk in response:
                    file.write(chunk)
            os.rename(tmp_dataset_file_path, dataset_file_path)
def make_envs_and_datasets(
    dataset_name,
    dataset_dir=DEFAULT_DATASET_DIR,
    compact_dataset=False,
    env_only=False,
    num_envs = 32,
    **env_kwargs,
):
    """ Exactly the same as make_env_and_datasets() from here https://github.com/seohongpark/ogbench/blob/master/ogbench/utils.py
    except this supports making vectorizing the envirnment so you don't have to run rollout evaluations sequentially. 
    """
    # Make environment.
    splits = dataset_name.split('-')
    env_name = '-'.join(splits[:-2] + splits[-1:])  # Remove the dataset type.
    env_make_list = [lambda: gymnasium.make(env_name, **env_kwargs)] * num_envs
    envs = gymnasium.vector.AsyncVectorEnv(env_make_list)
    #env = gymnasium.make(env_name, **env_kwargs) old single environment

    if env_only:
        return envs

    # Load datasets.
    dataset_dir = os.path.expanduser(dataset_dir)
    download_datasets([dataset_name], dataset_dir)
    train_dataset_path = os.path.join(dataset_dir, f'{dataset_name}.npz')
    val_dataset_path = os.path.join(dataset_dir, f'{dataset_name}-val.npz')
    ob_dtype = np.uint8 if ('visual' in env_name or 'powderworld' in env_name) else np.float32
    action_dtype = np.int32 if 'powderworld' in env_name else np.float32
    train_dataset = load_dataset(
        train_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=compact_dataset,
    )
    val_dataset = load_dataset(
        val_dataset_path,
        ob_dtype=ob_dtype,
        action_dtype=action_dtype,
        compact_dataset=compact_dataset,
    )

    return envs, train_dataset, val_dataset

class DecagonDataset():
    """
    Dataloader for contrastive classifier. 

    """
    def __init__(self, 
                 config,
                 give_t5_encoding = True):
        self.config = config
        self.batch_size = config.batch_size
        self.give_t5_encoding = give_t5_encoding

        self.data = np.load(config.dataset_dir,allow_pickle=True)
        self.traj_len = self.data[0]['trajectory'].shape[0]

        
        
    def sample(self, idxs=None, evaluation=False,pad=False):

        batch_size = self.config.batch_size
        # gets IDs of the trajectories to sample from
        sample_traj_idx = idxs
        if sample_traj_idx is None:
            sample_traj_idx = self.get_random_traj_idx(batch_size) # ints of size (batch_size)
        trajectories = self.data[sample_traj_idx] # should be np array w/dicts, each dict is language plan/trajectory
       
        
        # I know doing this sequentially is slow but it should be alaright
        batch_obs = []
        batch_lang = []
        batch_plansketch = []
        for i,trajectory in enumerate(trajectories): # iterates (batch_size) times
            batch_obs.append(trajectory['trajectory'])
            landmarks = trajectory['landmarks'].flatten()
            distances = trajectory['distances'].flatten()/512
            plan_sketch = np.concatenate((landmarks,distances))
            #print(plan_sketch.shape)
            batch_plansketch.append(plan_sketch) 
            
        batch_obs, batch_lang = np.array(batch_obs), np.array(batch_lang)
        
        
        if pad:  # meaning power of 2 padding
            # pads to the nearest power of 2 if wanted. Uses last goal for obs, and 0 action for action.
            # Note that this won't always produce valid actions - for now, the padded trajectories are only
            # being used over the states so padding actions is just for consistency, but keep it in mind.
            n_samples = batch_obs.shape[1]
            new_length = 2 ** (int(np.log2(n_samples )) + 1) # find next higher power of 2
            batch_obs = np.concatenate((batch_obs,np.ones_like(batch_obs)*batch_obs[:,-1:,:]),axis=1)[:,:new_length]

        batch_obs = torch.tensor(batch_obs,dtype=torch.float)
        if self.give_t5_encoding:
            batch_lang = torch.tensor(batch_lang,dtype=torch.float)
        #batch_plansketch = torch.tensor(batch_plansketch,dtype=torch.float)
        batch_plansketch = batch_plansketch
        # don't need to do anything to goal b/c its just a single item 

        # 
        batch = {'observations':batch_obs,
                 'goals':batch_lang,
                 'plansketch':batch_plansketch}

        return batch
    def get_random_traj_idx(self,batch_size):
        return np.random.randint(low = 0,high = len(self.data), size = batch_size)
    


        


class GCSequenceDataset():
    """Sequence dataset class for language goal-conditioned RL.

    
    """
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config
        self.sample_traj_len = config.horizon
        self.batch_size = config.batch_size

        # Making sure goal-sampling is kosher

        # turning dataset into proper trajectories 
        traj, traj_len = [], []
        data_ = defaultdict(list)
        for i in trange(dataset["observations"].shape[0], desc="Processing trajectories"):
            data_["observations"].append(dataset["observations"][i])
            #data_["actions"].append(dataset["actions"][i])
            data_["plan_enc"].append(dataset["plan_enc"][i])

            if dataset["terminals"][i]:
                episode_data = {k: np.array(v, dtype=np.float32) for k, v in data_.items()}
                traj.append(episode_data)
                traj_len.append(episode_data["actions"].shape[0])
                # reset trajectory buffer
                data_ = defaultdict(list)
        self.traj = traj
        self.traj_len = np.array(traj_len)
        
    def sample(self, idxs=None, evaluation=False,pad=False):

        batch_size = self.config.batch_size
        # gets IDs of the trajectories to sample from
        sample_traj_idx = idxs
        if sample_traj_idx is None:
            sample_traj_idx = self.get_random_traj_idx(batch_size) # ints of size (batch_size)

        # gets start and ending indices of sampled sub-trajectories from each trajectory
        sample_start_idx, sample_end_idx = self.get_random_start_end_idx(sample_traj_idx)

        # gets goals from each trajectory
        sample_goal_idx = self.get_random_goal_idx(sample_traj_idx,sample_start_idx,sample_end_idx)
        
        # I know doing this sequentially is slow but it should be alright
        batch_obs = []
        batch_acts = []
        batch_goals = []
        for i,traj_idx in enumerate(sample_traj_idx): # iterates (batch_size) times
            traj = self.traj[traj_idx]
            sample_acts = traj['actions'][sample_start_idx[i]:sample_end_idx[i]]
            sample_goal = traj['observations'][sample_goal_idx[i]]
            sample_obs = np.copy(traj['observations'])
            if self.config.goal_padding:
                sample_obs[sample_goal_idx[i]:] = sample_goal
            sample_obs = sample_obs[sample_start_idx[i]:sample_end_idx[i]]
            # pads to be length of desired trajectory if needed
            pad_diff = self.sample_traj_len - sample_obs.shape[0] 
            #print(pad_diff)
            if pad_diff > 0:
                sample_obs = np.pad(sample_obs,((0,pad_diff),(0,0)),mode='edge')
                sample_acts = np.pad(sample_acts,((0,pad_diff),(0,0)),mode='edge')
            #print(sample_acts.shape)
            batch_obs.append(sample_obs)
            batch_acts.append(sample_acts)
            batch_goals.append(sample_goal)
        batch_obs, batch_acts, batch_goals = np.array(batch_obs), np.array(batch_acts), np.array(batch_goals)
        
        
        
        if pad:  # meaning power of 2 padding
            # pads to the nearest power of 2 if wanted. Uses last goal for obs, and 0 action for action.
            # Note that this won't always produce valid actions - for now, the padded trajectories are only
            # being used over the states so padding actions is just for consistency, but keep it in mind.
            n_samples = batch_obs.shape[1]
            new_length = 2 ** (int(np.log2(n_samples )) + 1) # find next higher power of 2
            batch_obs = np.concatenate((batch_obs,np.ones_like(batch_obs)*batch_obs[:,-1:,:]),axis=1)[:,:new_length]
            batch_acts = np.concatenate((batch_acts,np.zeros_like(batch_acts)),axis=1)[:,:new_length]
        # don't need to do anything to goal b/c its just a single item 
        batch = {'observations':batch_obs,'actions':batch_acts,'goals':batch_goals}


        # NOTE - this next thing won't work yet. gotta adapt it. 
        if self.config.p_aug is not None and not evaluation:
            if np.random.rand() < self.config.p_aug:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])

        return batch

    def get_random_goal_idx(self,traj_idx,sample_start_idx,sample_end_idx):
        """ Gets goals from each desired trajectory, given its existing ending ID. 
        Didn't implement the random goal thing becuase it seems really stupid to me >:P"""
        
        traj_end_idx = self.traj_len[traj_idx]
        
        # determines where to start the goal sampling 
        goal_dist_start = 0 # starting index of the goal-distribution
        if self.config.goal_sample_start == 'start':
            goal_dist_start = sample_start_idx
        elif self.config.goal_sample_start == 'end':
            goal_dist_start = sample_end_idx
        else:
            assert False, 'ERROR: Sampling Start Option is Invalid!'

        # determines where to end the goal sampling 
        goal_dist_end = 0 # starting index of the goal-distribution
        if self.config.goal_sample_end == 'sample':
            goal_dist_end = np.minimum(sample_end_idx - 1,traj_end_idx - 1)
        elif self.config.goal_sample_end == 'trajectory':
            goal_dist_end = traj_end_idx - 1 
        else:
            assert False, 'ERROR: Sampling End Option is Invalid!'

        # determines how to sample the goal distribution
        if self.config.goal_sample_dist == 'geometric':
            traj_future_idx = np.minimum(goal_dist_start + np.random.geometric((1.0-self.config.goal_sample_gamma)*np.ones_like(goal_dist_start)),goal_dist_end)
        elif self.config.goal_sample_dist == 'uniform':
            traj_future_idx = np.random.uniform(low = goal_dist_start, high = goal_dist_end).astype(int)
        elif self.config.goal_sample_dist == 'end':
            traj_future_idx = goal_dist_end
        else:
            assert False, 'ERROR: Goal Sampling Distribution is Invalid!'
        return traj_future_idx

        


    def get_random_start_end_idx(self,traj_idx):
        """ Gets random start and end IDX per-given trajectory. 
        Start is always after the beginning, end is before the end.
        This means no padding is performed. """
        traj_idx_lens = self.traj_len[traj_idx] # length of trajectory at each idx, size (batch_size)
        sample_traj_length = self.sample_traj_len # sample length, size (1)
        if self.config.allow_partial_subsamples: # case where we sample from support of whole trajectory
            last_allowable_start = traj_idx_lens
        else: # case where we sample from support of trajectory s.t. sample will be whole subsample
            last_allowable_start = traj_idx_lens - sample_traj_length
        # could make more efficent
        start_idx = np.random.uniform(low = np.zeros_like(last_allowable_start), 
                                      high = last_allowable_start).astype(int) # starting idx for each traj, size (batch_size)
        end_idx = start_idx + sample_traj_length # ending idx for each traj, size (batch_size)
        return start_idx, end_idx

    def get_random_traj_idx(self,batch_size):
        return np.random.randint(low = 0,high = len(self.traj_len), size = batch_size)
    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


### this is all just stuff from the OGBench utilities file for reference, shouldn't be used
### in our stuff 
class Dataset(FrozenDict):
    """Dataset class.

    This class supports both regular datasets (i.e., storing both observations and next_observations) and
    compact datasets (i.e., storing only observations). It assumes 'observations' is always present in the keys. If
    'next_observations' is not present, it will be inferred from 'observations' by shifting the indices by 1. In this
    case, set 'valids' appropriately to mask out the last state of each trajectory.
    """

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        if 'valids' in self._dict:
            (self.valid_idxs,) = np.nonzero(self['valids'] > 0)

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        if 'valids' in self._dict:
            return self.valid_idxs[np.random.randint(len(self.valid_idxs), size=num_idxs)]
        else:
            return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if 'next_observations' not in result:
            result['next_observations'] = self._dict['observations'][np.minimum(idxs + 1, self.size - 1)]
        return result


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0


@dataclasses.dataclass
class GCDataset:
    """Dataset class for goal-conditioned RL.

    This class provides a method to sample a batch of transitions with goals (value_goals and actor_goals) from the
    dataset. The goals are sampled from the current state, future states in the same trajectory, and random states.
    It also supports frame stacking and random-cropping image augmentation.

    It reads the following keys from the config:
    - discount: Discount factor for geometric sampling.
    - value_p_curgoal: Probability of using the current state as the value goal.
    - value_p_trajgoal: Probability of using a future state in the same trajectory as the value goal.
    - value_p_randomgoal: Probability of using a random state as the value goal.
    - value_geom_sample: Whether to use geometric sampling for future value goals.
    - actor_p_curgoal: Probability of using the current state as the actor goal.
    - actor_p_trajgoal: Probability of using a future state in the same trajectory as the actor goal.
    - actor_p_randomgoal: Probability of using a random state as the actor goal.
    - actor_geom_sample: Whether to use geometric sampling for future actor goals.
    - gc_negative: Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as the reward.
    - p_aug: Probability of applying image augmentation.
    - frame_stack: Number of frames to stack.

    Attributes:
        dataset: Dataset object.
        config: Configuration dictionary.
        preprocess_frame_stack: Whether to preprocess frame stacks. If False, frame stacks are computed on-the-fly. This
            saves memory but may slow down training.
    """

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = False

    def __post_init__(self):
        self.size = self.dataset.size

        # Pre-compute trajectory boundaries.
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        # Assert probabilities sum to 1.
        assert np.isclose(
            self.config.value_p_curgoal + self.config.value_p_trajgoal + self.config.value_p_randomgoal, 1.0
        )
        assert np.isclose(
            self.config.actor_p_curgoal + self.config.actor_p_trajgoal + self.config.actor_p_randomgoal, 1.0
        )

        if self.config.frame_stack is not None:
            # Only support compact (observation-only) datasets.
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(np.arange(self.size))
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals (value_goals and actor_goals) from the dataset. They are
        stored in the keys 'value_goals' and 'actor_goals', respectively. It also computes the 'rewards' and 'masks'
        based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config.frame_stack is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        value_goal_idxs = self.sample_goals(
            idxs,
            self.config.value_p_curgoal,
            self.config.value_p_trajgoal,
            self.config.value_p_randomgoal,
            self.config.value_geom_sample,
        )
        actor_goal_idxs = self.sample_goals(
            idxs,
            self.config.actor_p_curgoal,
            self.config.actor_p_trajgoal,
            self.config.actor_p_randomgoal,
            self.config.actor_geom_sample,
        )

        batch['value_goals'] = self.get_observations(value_goal_idxs)
        batch['actor_goals'] = self.get_observations(actor_goal_idxs)
        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config.gc_negative else 0.0)

        if self.config.p_aug is not None and not evaluation:
            if np.random.rand() < self.config.p_aug:
                self.augment(batch, ['observations', 'next_observations', 'value_goals', 'actor_goals'])

        return batch

    def sample_goals(self, idxs, p_curgoal, p_trajgoal, p_randomgoal, geom_sample):
        """Sample goals for the given indices."""
        batch_size = len(idxs)

        # Random goals.
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)

        # Goals from the same trajectory (excluding the current state, unless it is the final state).
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        if geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config.discount, size=batch_size)  # in [1, inf)
            middle_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            middle_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        goal_idxs = np.where(
            np.random.rand(batch_size) < p_trajgoal / (1.0 - p_curgoal + 1e-6), middle_goal_idxs, random_goal_idxs
        )

        # Goals at the current state.
        goal_idxs = np.where(np.random.rand(batch_size) < p_curgoal, idxs, goal_idxs)

        return goal_idxs

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )

    def get_observations(self, idxs):
        """Return the observations for the given indices."""
        if self.config.frame_stack is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: arr[idxs], self.dataset['observations'])
        else:
            return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        """Return the frame-stacked observations for the given indices."""
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(self.config.frame_stack)):
            cur_idxs = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)


@dataclasses.dataclass
class HGCDataset(GCDataset):
    # NOTE - we never use this but it's easier to keep for compatability
    """Dataset class for hierarchical goal-conditioned RL.

    This class extends GCDataset to support high-level actor goals and prediction targets. It reads the following
    additional key from the config:
    - subgoal_steps: Subgoal steps (i.e., the number of steps to reach the low-level goal).
    """

    def sample(self, batch_size: int, idxs=None, evaluation=False):
        """Sample a batch of transitions with goals.

        This method samples a batch of transitions with goals from the dataset. The goals are stored in the keys
        'value_goals', 'low_actor_goals', 'high_actor_goals', and 'high_actor_targets'. It also computes the 'rewards'
        and 'masks' based on the indices of the goals.

        Args:
            batch_size: Batch size.
            idxs: Indices of the transitions to sample. If None, random indices are sampled.
            evaluation: Whether to sample for evaluation. If True, image augmentation is not applied.
        """
        if idxs is None:
            idxs = self.dataset.get_random_idxs(batch_size)

        batch = self.dataset.sample(batch_size, idxs)
        if self.config.frame_stack is not None:
            batch['observations'] = self.get_observations(idxs)
            batch['next_observations'] = self.get_observations(idxs + 1)

        # Sample value goals.
        value_goal_idxs = self.sample_goals(
            idxs,
            self.config.value_p_curgoal,
            self.config.value_p_trajgoal,
            self.config.value_p_randomgoal,
            self.config.value_geom_sample,
        )
        batch['value_goals'] = self.get_observations(value_goal_idxs)

        successes = (idxs == value_goal_idxs).astype(float)
        batch['masks'] = 1.0 - successes
        batch['rewards'] = successes - (1.0 if self.config.gc_negative else 0.0)

        # Set low-level actor goals.
        final_state_idxs = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        low_goal_idxs = np.minimum(idxs + self.config.subgoal_steps, final_state_idxs)
        batch['low_actor_goals'] = self.get_observations(low_goal_idxs)

        # Sample high-level actor goals and set prediction targets.
        # High-level future goals.
        if self.config.actor_geom_sample:
            # Geometric sampling.
            offsets = np.random.geometric(p=1 - self.config.discount, size=batch_size)  # in [1, inf)
            high_traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            # Uniform sampling.
            distances = np.random.rand(batch_size)  # in [0, 1)
            high_traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        high_traj_target_idxs = np.minimum(idxs + self.config.subgoal_steps, high_traj_goal_idxs)

        # High-level random goals.
        high_random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        high_random_target_idxs = np.minimum(idxs + self.config.subgoal_steps, final_state_idxs)

        # Pick between high-level future goals and random goals.
        pick_random = np.random.rand(batch_size) < self.config.actor_p_randomgoal
        high_goal_idxs = np.where(pick_random, high_random_goal_idxs, high_traj_goal_idxs)
        high_target_idxs = np.where(pick_random, high_random_target_idxs, high_traj_target_idxs)

        batch['high_actor_goals'] = self.get_observations(high_goal_idxs)
        batch['high_actor_targets'] = self.get_observations(high_target_idxs)

        if self.config.p_aug is not None and not evaluation:
            if np.random.rand() < self.config.p_aug:
                self.augment(
                    batch,
                    [
                        'observations',
                        'next_observations',
                        'value_goals',
                        'low_actor_goals',
                        'high_actor_goals',
                        'high_actor_targets',
                    ],
                )

        return batch
    
