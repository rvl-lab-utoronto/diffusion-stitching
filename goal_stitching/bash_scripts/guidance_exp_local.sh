### Training models for the experiments w/diffusion guidance

## Scene environment

# Janner Diffuser (unconditional w/impainting)
# python janner_diffuser.py \
# --use_wandb True \
# --name 'JannerInpaint' \
# --env 'scene-play-v0' \
# --checkpoints_path 'trained_models' \
# --eval_interval 100000000 \
# --invdyn_dir 'common_models' \
# --horizon 512 \
# --batch_size 128 \
# --gradient_steps 500000 \
# --log_interval 100 \
# --save_interval 100000 \
# --num_episodes 1

# Goal-Conditioned Diffuser
# python decision_diffuser.py \
# --use_wandb True \
# --name 'GCDiffuser' \
# --env 'scene-play-v0' \
# --checkpoints_path 'trained_models' \
# --eval_interval 100000000 \
# --invdyn_dir 'common_models' \
# --horizon 512 \
# --batch_size 128 \
# --gradient_steps 500000 \
# --log_interval 100 \
# --save_interval 100000 \
# --num_episodes 1 \
# --label_dropout 0 \
# --w_cfg 0.0

# Decision Diffuser
python decision_diffuser.py \
--use_wandb True \
--name 'CFGDiffuser' \
--env 'scene-play-v0' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 512 \
--batch_size 128 \
--gradient_steps 500000 \
--log_interval 100 \
--save_interval 100000 \
--num_episodes 1 \
--label_dropout 0.25