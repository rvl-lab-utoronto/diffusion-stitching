"""
NOTE - this is just meant for use with the language experiments. Won't work generally and lots
of stuff is badly hard coded - see the goal_stitching section if you want to use a more general

"""

import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import uuid
import pyrallis
from tqdm import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR
from dataclasses import asdict, dataclass
from utilities.cleandiffuser.diffusion import DiscreteDiffusionSDE
from utilities.cleandiffuser.nn_diffusion import JannerUNet1d,CNN1dShiftEq,DiT1d
from utilities.cleandiffuser.classifier import CumRewClassifier
from utilities.cleandiffuser.nn_classifier import HalfJannerUNet1d
from utilities.cleandiffuser.nn_condition import MLPCondition
from utilities.cleandiffuser.utils import report_parameters
from utilities.ogbench_utilities import set_seed
from utilities.cleandiffuser.diffusion import ddpm
from utilities.cleandiffuser.invdynamic import MlpInvDynamic
from typing import Optional
from utilities.ogbench_utilities import *
from utilities.decagon_env import *
from utilities.toy_env_utilities import ToyEnvInvDyn
from transformers import T5Tokenizer, T5Model, T5TokenizerFast


@dataclass
class DiffusionPlannerConfig:

    ### Logging w/WanDB
    use_wandb: bool = False
    project: str = "data"
    group: str = "decagon"
    name: str = "DP"
    log_interval: int = 100

    ### Environment
    # training dataset and evaluation environment
    env: str = 'decagon' 
    # path for checkpoints saving, optional
    checkpoints_path: Optional[str] = 'trained_models'
    # file name for loading a model, optional
    load_model: str = ""
    # training random seed
    seed: int = 0
    # training device
    device: str = "cuda"

    ### Diffusion Planner Architecture/Training Parameters
    # horizon to plan over during training
    horizon: int =  512 # must be power of 2 
    # dimension of model's internal representaitons
    model_dim: int =  64
    # number of forward diffusion steps to take during training
    diffusion_steps: int = 100
    lang_enc_size: int = 768    
    # whether to predict noise or a clean image
    predict_noise: bool =  False
    # whether to use self-attention. False in default implementation
    self_attention: bool = False 
    # whether to use transformer architecture. false by defalt
    use_transformer: bool = False
    # whether to use positional embeddings for the time in the denoising process. True in default implementation
    use_timestep_embeddings: bool = True 
    # whether to use shift equivariant architecture
    use_shift_equivariant_arch: bool = False 
    # channel size for equivariant net
    channel_size: int = 64
    # whether to add positional encoding for the position in the sequence to each layer
    add_positional_encoding: bool = False

    # if using EqNet, kernel expansion rate
    kernel_expansion_rate: int = 5 # usually strikes best balance at depth = 25
    # unused
    action_loss_weight: float =  10. # shouldn't be used

    ### Guidance Options
    # what form of guidance to use. Current options - none, cfg, classifier. for multiple, put both in string.
    guidance: str = 'cfg'
    # if using classifier guidance, value of weight to apply to gradient.
    w_cg: float =  0.001
    # if using CFG, weight of CFG. see original paper for description
    w_cfg: float = 1.0
    # hidden layer size of classifier
    emb_dim: int = 64
    # label dropout for CFG. Called p_uncond in the original CFG paper
    label_dropout: float = 0.25

    ### Traditional ML Training Parameters
    # batch size
    batch_size: int = 128
    # number of gradient steps to take
    gradient_steps: int = 200_000
    # how frecnetly to evaluate the model
    eval_interval: int = 99999999999999999999
    # how frequently to save the model
    save_interval: int = 50_000

    dataset_dir: str = 'decagon_data/train_data.npy'

    ### Evaluation Parameters
    ckpt: str = 'latest'
    # solver to use during inference
    solver: str = 'ddpm'
    # number of reverse diffusion steps to take during inference
    sampling_steps: int =  100
    # number of states kept "in memory" during replanning
    memory: int = 20  
    # number of environments to run in parralel
    num_envs: int =  1 
    # number of episodes to run during evaluation, across num_env parralel environments
    num_episodes: int =  10 
    # stochastic temperature of sampling
    temperature: float =  0.5
    # whether to use EMA for evaluation weights
    use_ema: bool =  True
    # expontential moving average rate for previous parameters
    ema_rate: float =  0.9999
    # number of steps before environment times out
    env_timeout_steps: int = 1000
    # number of steps to execute in the environment
    n_exec_steps: int = 450
    # directory for inverse dynamics model, if required (for non-toy environments)
    invdyn_dir: str = 'common_models' # directory where the IK model is stored
    # adds goal inpainting
    inpaint: bool = True 
    # how many steps at the end to inpaint
    goal_inpaint_steps: int = 25
    # whether to perform the inverse dynamics over a plan open or closed loop. True typically works better.
    open_loop_invdyn: bool = True 


    pad:bool = False

    #### Dataset hyparparameters used by the sequence dataloader. 
    ### sequence start options
    # whether to require the sample to be entirely within the sequence or to allow sub-sampling w/padding
    
    

    ### Dataset hyperparameters needed for compatability with OGBench. Should not do anything. Try not to change.
    dataset_class='GCDataset'  # Dataset class name.
    # btw, NONE of the next stuff gets used here b/c it just conditions
    # with inpainting. 
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

    # should all be unused
    normalize: bool = True # state normalization
    discount: float = 0.99

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)
        self.invdyn_path = self.invdyn_dir + '/' + self.env + '-invdyn.pt'
        
        # checks stuff
        assert self.n_exec_steps <= self.horizon - self.memory # 

        self.flatland = 'flatland' in self.env
        self.gridland = 'gridland' in self.env
        self.toy_env = self.flatland or self.gridland
        if self.flatland:
            self.n_dims = int(self.env[self.env.find('-n')+2])
        if self.gridland:
            self.n_size = int(self.env[self.env.find('-n')+2])
        if self.toy_env:
            self.gc_dataset = 'gc' in self.env # determines if goal-conditioned ataset should be used
        if not self.pad:
            self.gen_horizon = self.horizon
        else:
            self.gen_horizon = 2 ** (int(np.log2(self.horizon )) + 1) # find next higher power of 2

        

@pyrallis.wrap()
def train(config: DiffusionPlannerConfig):
    if config.use_wandb:
        wandb_init(asdict(config))
    set_seed(config.seed)
    if config.checkpoints_path is not None:
        if os.path.exists(config.checkpoints_path) is False:
            os.makedirs(config.checkpoints_path)

    # ---------------------- Create Dataset ----------------------
    
    
    train_dataset = DecagonDataset(config) # new sequence dataset
    envs, _, _ = make_envs_decagon(num_envs = config.num_envs)
    obs_dim, act_dim = envs.single_observation_space.shape[0], envs.single_action_space.shape[0]

    # --------------- Network Architecture -----------------
    if config.use_shift_equivariant_arch:
        #nn_diffusion = ConvNext1dShiftEq(obs_dim)
        nn_diffusion = CNN1dShiftEq(obs_dim,
                                    kernel_expansion_rate=config.kernel_expansion_rate,
                                    model_dim = config.model_dim,
                                    emb_dim = config.emb_dim,
                                    encode_position = config.add_positional_encoding,
                                    channel_size=config.channel_size)
    elif config.use_transformer:
        nn_diffusion  = DiT1d(
                            obs_dim, 
                            emb_dim=config.model_dim,
                            d_model=256, 
                            n_heads=4, 
                            depth=4, 
                            timestep_emb_type="fourier")
    else:
        nn_diffusion = JannerUNet1d(
            obs_dim, model_dim=config.model_dim, emb_dim=config.model_dim, dim_mult=[1, 2, 2, 2],
            timestep_emb_type="positional", attention=config.self_attention, kernel_size=5,
            use_timestep_emb=config.use_timestep_embeddings)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"======================= Parameter Report of Classifier =======================")
    # creates classifier network if needed
    if 'classifier' in config.guidance:
        nn_classifier = HalfJannerUNet1d(
            config.gen_horizon, obs_dim + act_dim, out_dim=1,
            model_dim=config.model_dim, emb_dim=config.model_dim, dim_mult=config.task.dim_mult,
            timestep_emb_type="positional", kernel_size=3)
        classifier = CumRewClassifier(nn_classifier, device=config.device)
    else:
        classifier = None
    # creates condition network if needed (when using classifier-free guidance)
    if 'cfg' in config.guidance:
        nn_condition = MLPCondition(
        in_dim=config.lang_enc_size, out_dim=config.emb_dim, hidden_dims=[config.emb_dim, ], act=nn.SiLU(), dropout=config.label_dropout)
    else:
        nn_condition = None
    # ----------------- Masking ------------------- # some changes to remove gen. over actions
    # fix_mask that tells Diffusion model what parts of sequence to ignore during training
    # and which parts to inpaint with a prior during sampling. We'll only mask the first
    # state now (which should never be getting predicted from random)
    # but all others we keep during training, and then modify later during sampling.
    fix_mask = torch.zeros((config.gen_horizon, obs_dim)) 
    #fix_mask[0, :] = 1. # for the starting/current state
    loss_weight = torch.ones((config.gen_horizon, obs_dim))

    # --------------- Diffusion Model --------------------
    agent = DiscreteDiffusionSDE(
        nn_diffusion = nn_diffusion, 
        nn_condition = nn_condition,
        classifier = classifier, 
        fix_mask=fix_mask, 
        loss_weight=loss_weight, 
        ema_rate=config.ema_rate,
        device=config.device,
        diffusion_steps=config.diffusion_steps, 
        predict_noise=config.predict_noise)
    
    # --------------- Inverse Dynamics --------------------
    invdyn = ToyEnvInvDyn()

    # ---------------------- Training ----------------------

    diffusion_lr_scheduler = CosineAnnealingLR(agent.optimizer, config.gradient_steps)

    agent.train()

    n_gradient_step = 0
    log = {"avg_loss_diffusion": 0., "avg_loss_classifier": 0.}

    for i in tqdm(range(1, config.gradient_steps + 1), smoothing=0.1, dynamic_ncols=True):

        batch = train_dataset.sample(pad = config.pad) # one weird thing is we don't pass batch size
        obs = torch.tensor(batch["observations"],device=config.device)
        goal = torch.tensor(batch['goals'],device=config.device) if 'none' not in config.guidance else None
        x = obs
        # ----------- Gradient Step ------------
        
        log["avg_loss_diffusion"] += agent.update(x, goal)['loss']
        diffusion_lr_scheduler.step()

        # ----------- Evaluation ------------ # 
        
        if (n_gradient_step + 1) % config.eval_interval == 0:
            # Runs eval
            print('==== Running Eval ====')
            for task_id in [1,2,3,4,5]:
                print('== Task',task_id,' ==')
                envs.reset(options = {'task_id':task_id})
                avg_completion = eval_model(agent,invdyn,envs,config)
                if config.use_wandb:
                    wandb.log({"avg_completion_task_" + str(task_id): avg_completion,},step=i,)

        # ----------- Logging ------------ # 
        if (n_gradient_step + 1) % config.log_interval == 0:
            log["gradient_steps"] = n_gradient_step + 1
            log["avg_loss_diffusion"] /= config.log_interval
            if config.use_wandb:
                wandb.log({
                        "avg_loss_diffusion": log["avg_loss_diffusion"],
                        'learning_rate': diffusion_lr_scheduler.get_last_lr()[0]
                        },step=i,)
            log = {"avg_loss_diffusion": 0.}

        # ----------- Saving ------------
        
        if config.checkpoints_path is not None:
            if (n_gradient_step + 1) % config.save_interval == 0:
                agent.save(config.checkpoints_path + f"diffusion_ckpt_{n_gradient_step + 1}.pt")
                agent.save(config.checkpoints_path + f"diffusion_ckpt_latest.pt")

        n_gradient_step += 1

    
def eval_model(agent,invdyn,envs,config,give_trajectory = False,give_replans = False):
# ---------------------- Inference ----------------------

    agent.eval()
    # resets inpainting mask
    fix_mask = agent.fix_mask
    fix_mask = fix_mask * 0.0
    fix_mask[:,0, :] = 1. # for the starting/current state
    if config.inpaint:
        fix_mask[:,-config.goal_inpaint_steps:, :] = 1.
    agent.fix_mask = fix_mask # for ending state

    #normalizer = dataset.get_normalizer()
    avg_completion = []
    obs_dim = envs.single_observation_space.shape[0]
    prior = torch.zeros((config.num_envs, config.gen_horizon, obs_dim), device=config.device) # size (num_envs, config.horizon, obs_dim)
    obs_recording = []
    replans = []
    for i in trange(config.num_episodes):
        task_id = envs.get_attr('cur_task_id')[0]
        obs, info = envs.reset(options = {'task_id': task_id})
        #obs[:,1] = 5.0
        # TODO change back for testing
        t = 0.
        completed = np.zeros((config.num_envs)).astype(bool) # - per-env completion
        goals = torch.tensor(info['goal'],device=config.device).to(torch.float) # Goal state - shape (n_envs, obs_dim)
        #goals[:,0], goals[:,1] = 0, 35 # TODO change back for testing
        #print(goals)
        truncated = False
        state_history = [obs]
        #fig, ax = plt.subplots(figsize=(6, 6))
        while not np.all(completed) and t < config.env_timeout_steps + 1 and (not np.any(truncated)):

            # sample trajectories
            # inpaints history to beginning of generated trajectory
            history_len = min(len(state_history),config.memory)
            state_history_array = torch.tensor(np.array(state_history),device=config.device) # shape (len(state_history), n_envs, obs_dim)
            state_history_array = state_history_array[-history_len:] # shape (history_len,n_envs,obs_dim)
            state_history_array = torch.transpose(state_history_array,0,1)
            prior[:, :history_len, :] = state_history_array#[:,-history_len:,:] # inpaints start state at beginning of trajectory
            # modifies agent mask to account for history
            fix_mask = agent.fix_mask 
            fix_mask[:,:history_len, :] = 1. 
            #fix_mask = fix_mask *0 # TODO change back 
            agent.fix_mask = fix_mask

            # inpaints goal state if desired
            condition_cfg = None
            if config.inpaint:
                # TODO change this back
                prior[:, -1, :obs_dim] = torch.tensor(goals,device=config.device) # inpaints goal state at end of trajectory
                #prior[:, -config.goal_inpaint_steps:, :obs_dim] = goals.clone().detach()
                prior[:, -config.goal_inpaint_steps:, :obs_dim] = torch.unsqueeze(goals.clone().detach(),dim = 1)

            if 'cfg' in config.guidance:
                condition_cfg = goals
            
            traj, log = agent.sample(
                prior,
                solver='ddpm',
                n_samples=config.num_envs,
                sample_steps=config.sampling_steps,
                use_ema=config.use_ema, 
                w_cg=config.w_cg, 
                temperature=config.temperature,
                condition_cfg=condition_cfg)
            replans.append(traj.cpu().numpy())
            #print(traj[0,1:,:])
            # run inverse dynamics model to generate action plan=
            with torch.no_grad():
                # invdyn goes predict(current_state,next_state)
                open_loop_acts = invdyn.predict(traj[:, :-1, :], traj[:, 1:, :]).cpu().numpy() # shape (n_envs,horizon_len,act_size)
            # cuts out actions before the history horizon
            open_loop_acts = open_loop_acts[:,history_len-1:]
            # steps though appropriate number of action steps 
            curr_act_idx = 0
            # next line is just "if we're not done and don't want to replan yet"
            #print('meow')
            while (not np.all(completed) and t < config.env_timeout_steps + 1) and (curr_act_idx < config.n_exec_steps) and (not np.any(truncated)):
                # terminated gives True if that env. has been successfully completed
                # truncated gives True if that env. has reached timeout limit (all envs will do this at once)
                if config.open_loop_invdyn: # run through whole action plan before replanning
                    obs, rew, terminated, truncated, info = envs.step(open_loop_acts[:,curr_act_idx,:])
                else: # re-run invdyn to go from current state -> next desired state in plan
                    # obs - shape (num_envs, 1, obs_dim)
                    act = invdyn.predict(torch.tensor(obs,device=config.device),traj[:,history_len+curr_act_idx]).cpu().numpy()
                    obs, rew, terminated, truncated, info = envs.step(act)
                curr_act_idx += 1
                t += 1
                #print(t)
                completed = np.logical_or(completed, terminated)
                state_history.append(obs)
            
        avg_completion.append(np.mean(completed))
        obs_recording.append(np.array(state_history))

    agent.train()
    return_list = [np.mean(np.array(avg_completion))]
    if give_trajectory:
        return_list.append(obs_recording)
    if give_replans:
        return_list.append(np.transpose(np.array(replans),(1,0,2,3)))
        #return_list.append(replans)
    
    return tuple(return_list)

def get_trajectory(agent,config,condition=None):
    if condition is not None:
        if type(condition) is str:
            # it's language so pass it through the T5 encoding thing
            device = 'cuda'
            tokenizer = T5TokenizerFast.from_pretrained("t5-base")
            text_model = T5Model.from_pretrained("t5-base").to(device)
            input_ids = tokenizer(
                "", return_tensors="pt"
            ).input_ids.to(device)  # Batch size 1
            decoder_input_ids = tokenizer(condition, return_tensors="pt").input_ids.to(device)  # Batch size 1
            outputs = text_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            last_hidden_states =  torch.squeeze(torch.mean(outputs.last_hidden_state,dim=1))
            condition = last_hidden_states
            #print('Condition Encoded Correctly')
        else: # assume it's a numpy array -> convert to tensor
            condition = torch.tensor(condition,device=config.device)
            pass 

    agent.eval()
    prior = torch.zeros((config.num_envs, config.gen_horizon, 2), device=config.device) # size (num_envs, config.horizon, obs_dim)
    replans = []
    for i in range(config.num_episodes):
            # inpaints goal state if desired
            condition_cfg = None
            
            if 'cfg' in config.guidance:
                #print('Condition Loaded Correctly')
                condition_cfg = condition
            
            traj, log = agent.sample(
                prior,
                solver='ddpm',
                n_samples=config.num_envs,
                sample_steps=config.sampling_steps,
                use_ema=config.use_ema, 
                w_cg=config.w_cg, 
                w_cfg = config.w_cfg,
                temperature=config.temperature,
                condition_cfg=condition_cfg)
            replans.append(traj.cpu().numpy())
            
    #replans = (np.transpose(np.array(replans),(1,0,2,3))) # not 100% what this does... whatever
        #return_list.append(replans)
    
    return replans
    

if __name__ == "__main__":
    train()
