""" This just trains the inverse dynamics model for each environment. Pretty
simple. """
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import pyrallis
from dataclasses import asdict, dataclass
from utilities.cleandiffuser.invdynamic import MlpInvDynamic
from utilities.ogbench_utilities import *
import tqdm

@dataclass
class InvDynConfig:
    env: str = 'scene-play-v0'
    # total gradient updates during training
    gradient_steps: int = int(1e5)
    # training batch size
    batch_size: int = 2048
    # whether to normalize states
    normalize: bool = True
    # path for checkpoints saving, optional
    checkpoints_path: str = 'common_models'
    # file name for loading a model, optional
    load_model: str = ""
    # training random seed
    seed: int = 0
    # training device
    device: str = "cuda"
    
    # Dataset hyperparameters needed for compatability with OGBench. 
    # Try not to change but literally none of this should be used.

    dataset_class='GCDataset'  # Dataset class name.
    value_p_curgoal=0.0  # Unused (defined for compatibility with GCDataset).
    value_p_trajgoal=1.0  # Unused (defined for compatibility with GCDataset).
    value_p_randomgoal=0.0  # Unused (defined for compatibility with GCDataset).
    value_geom_sample=False  # Unused (defined for compatibility with GCDataset).
    actor_p_curgoal=0.0  # Probability of using the current state as the actor goal.
    actor_p_trajgoal=1.0  # Probability of using a future state in the same trajectory as the actor goal.
    actor_p_randomgoal=0.0  # Probability of using a random state as the actor goal.
    actor_geom_sample=False  # Whether to use geometric sampling for future actor goals.
    frame_stack= None
    gc_negative=True  # Unused (defined for compatibility with GCDataset).
    p_aug=0.0  # Probability of applying image augmentation.

    def __post_init__(self):
        self.name = self.env
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)

@pyrallis.wrap()
def pipeline(config: InvDynConfig):
    env, train_dataset = load_ogbench_data_env(config.env,config)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.close() # don't need it 

    # --------------- Inverse Dynamic -------------------
    invdyn = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=config.device)

    # ---------------------- Training ----------------------

    invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, config.gradient_steps)

    invdyn.train()

    for i in tqdm.tqdm(range(1, config.gradient_steps + 1), smoothing=0.1, dynamic_ncols=True):
        
        batch = train_dataset.sample(config.batch_size)

        obs = torch.tensor(batch["observations"],device=config.device)
        next_obs = torch.tensor(batch["next_observations"],device=config.device)
        act = torch.tensor(batch["actions"],device=config.device)

        # ----------- Gradient Step ------------
        loss = invdyn.update(obs, act, next_obs)['loss']
        invdyn_lr_scheduler.step()

        if i % 10000 == 0:
            print('Loss:',loss)


    invdyn.save(config.checkpoints_path + f"-invdyn.pt")


if __name__ == "__main__":
    pipeline()