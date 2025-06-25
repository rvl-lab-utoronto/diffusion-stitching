### Training models for the experiments w/diffusion guidance

## Scene environment
#Janner Diffuser (unconditional w/impainting)
python janner_diffuser.py \
--use_wandb True \
--name 'JannerInpaint' \
--env 'flatland2' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 128 \
--batch_size 128 \
--gradient_steps 50000 \
--log_interval 100 \
--save_interval 50000 \
--num_episodes 1 \
--flatland True \
--n_dims 2 \
--n_exec_steps 1

# Goal-Conditioned Diffuser
python decision_diffuser.py \
--use_wandb True \
--name 'GCDiffuser' \
--env 'flatland2' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 128 \
--batch_size 128 \
--gradient_steps 50000 \
--log_interval 100 \
--save_interval 50000 \
--num_episodes 1 \
--label_dropout 0 \
--w_cfg 0.0 \
--flatland True \
--n_dims 2 \
--n_exec_steps 1

# Decision Diffuser
python decision_diffuser.py \
--use_wandb True \
--name 'CFGDiffuser' \
--env 'flatland2' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 128 \
--batch_size 128 \
--gradient_steps 50000 \
--log_interval 100 \
--save_interval 50000 \
--num_episodes 1 \
--label_dropout 0.25 \
--flatland True \
--n_dims 2 \
--n_exec_steps 1