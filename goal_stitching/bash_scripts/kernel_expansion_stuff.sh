### Training models for the single goal generation experiments

# all are N=5 gridland

### below - predicting clean
# shift equivariant training test
python diffusion_planner.py \
--use_wandb True \
--project 'composition' \
--group 'GridLand' \
--name 'DP-EqNetSmall-Uncond-FullSeq-UncondEnv-kexp5' \
--env 'gridland-n5' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 500 \
--batch_size 128 \
--gradient_steps 150000 \
--log_interval 100 \
--save_interval 150000 \
--num_episodes 1 \
--n_exec_steps 1 \
--goal_padding False \
--predict_noise False \
--pad True \
--self_attention False \
--allow_partial_subsamples False \
--use_timestep_embeddings True \
--use_shift_equivariant_arch True \
--guidance 'none' \
--goal_sample_dist 'end' \
--kernel_expansion_rate 5

python diffusion_planner.py \
--use_wandb True \
--project 'composition' \
--group 'GridLand' \
--name 'DP-EqNetSmall-Uncond-FullSeq-UncondEnv-kexp8' \
--env 'gridland-n5' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 500 \
--batch_size 128 \
--gradient_steps 150000 \
--log_interval 100 \
--save_interval 150000 \
--num_episodes 1 \
--n_exec_steps 1 \
--goal_padding False \
--predict_noise False \
--pad True \
--self_attention False \
--allow_partial_subsamples False \
--use_timestep_embeddings True \
--use_shift_equivariant_arch True \
--guidance 'none' \
--goal_sample_dist 'end' \
--kernel_expansion_rate 8

