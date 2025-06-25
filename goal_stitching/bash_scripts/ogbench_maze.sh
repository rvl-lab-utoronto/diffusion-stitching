### Training models for the single goal generation experiments

# all are N=5 gridland

### below - predicting clean
# shift equivariant training test

python diffusion_planner.py \
--use_wandb True \
--project 'composition' \
--group 'PointMaze' \
--name 'DP-UNet' \
--env 'pointmaze-giant-navigate-v0' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 2000 \
--batch_size 32 \
--gradient_steps 500000 \
--log_interval 100 \
--save_interval 100000 \
--num_episodes 1 \
--n_exec_steps 1 \
--predict_noise False \
--pad True \
--self_attention False \
--allow_partial_subsamples False \
--use_timestep_embeddings True \
--use_shift_equivariant_arch False \
--guidance 'none' \
--goal_sample_dist 'uniform' \
--goal_padding False \
--num_envs 1 \











