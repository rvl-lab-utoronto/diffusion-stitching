import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from ogbench import *
from utilities.ogbench_utilities import *

TensorBatch = List[torch.Tensor]


@dataclass
class GCBCTrainConfig:
    # wandb project name
    project: str = "goal-stitch"
    # wandb group name
    group: str = "OGBench"
    # wandb run name
    name: str = "GCBC"
    # training dataset and evaluation environment
    env: str = "antmaze-medium-stitch-v0"
    # total gradient updates during training
    max_timesteps: int = int(1e6)
    # training batch size
    batch_size: int = 256
    # maximum size of the replay buffer
    buffer_size: int = 2_000_000
    # what top fraction of the dataset (sorted by return) to use
    frac: float = 0.1
    # maximum trajectory length trained on 
    sample_traj_len: int = 256
    # whether to normalize states
    normalize: bool = True
    # discount factor
    discount: float = 0.99
    # evaluation frequency, will evaluate eval_freq training steps
    eval_freq: int = int(5e3)
    # number of episodes to run during evaluation
    n_episodes: int = 10
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = None
    # file name for loading a model, optional
    load_model: str = ""
    # training random seed
    seed: int = 0
    # training device
    device: str = "cuda"
    # Dataset hyperparameters needed for compatability with OGBench. Try not to change.
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
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)



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


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)

class Actor(nn.Module):
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class GCBC:
    def __init__(
        self,
        max_action: np.ndarray,
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        discount: float = 0.99,
        device: str = "cpu",
    ):
        self.actor = actor
        self.actor_optimizer = actor_optimizer
        self.max_action = max_action
        self.discount = discount

        self.total_it = 0
        self.device = device

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1

        state, action, _, _, _ = batch

        # Compute actor loss
        pi = self.actor(state)
        actor_loss = F.mse_loss(pi, action)
        log_dict["actor_loss"] = actor_loss.item()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.total_it = state_dict["total_it"]


@pyrallis.wrap()
def train(config: GCBCTrainConfig):
    env, train_dataset = load_ogbench_data_env(config.env,config)


    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]



    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    max_action = float(env.action_space.high[0])

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "device": config.device,
    }

    print("---------------------------------------")
    print(f"Training BC, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize policy
    trainer = GCBC(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    wandb_init(asdict(config))

    evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = train_dataset.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f}"
            )
            print("---------------------------------------")

            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

            wandb.log(
                {"eval_score": eval_score},
                step=trainer.total_it,
            )


if __name__ == "__main__":
    train()