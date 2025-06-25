### Training models for toy crosshatch appendix experiment

# Eq-Net, Unconditional, No Aug

python diffusion_planner.py \
--use_wandb True \
--project 'comp-paper' \
--group 'GridLand' \
--name 'DP-EqNet-KL' \
--env 'gridland-n5' \
--checkpoints_path 'appendix_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 500 \
--batch_size 128 \
--gradient_steps 100000 \
--log_interval 100 \
--save_interval 50000 \
--num_episodes 1 \
--n_exec_steps 1 \
--goal_padding False \
--predict_noise False \
--pad True \
--self_attention False \
--allow_partial_subsamples False \
--use_shift_equivariant_arch True \
--add_positional_encoding False \
--guidance 'none' \
--goal_sample_dist 'end' \
--override_loader True \
--override_path 'utilities/gridland_data_new/crosshatch.npy'
