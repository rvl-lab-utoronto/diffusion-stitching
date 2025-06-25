### Training models for the single goal generation experiments

# all are N=5 gridland

### below - predicting clean
# full trajectory training
python janner_diffuser.py \
--use_wandb True \
--name 'JannerInpaint-hFull-Clean' \
--env 'gridland-n5' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 500 \
--batch_size 128 \
--gradient_steps 500000 \
--log_interval 100 \
--save_interval 500000 \
--num_episodes 1 \
--n_size 5 \
--n_exec_steps 1 \
--goal_padding False \
--predict_noise False \
--pad True

# ~half trajectory
python janner_diffuser.py \
--use_wandb True \
--name 'JannerInpaint-h256-Clean' \
--env 'gridland-n5' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 256 \
--batch_size 128 \
--gradient_steps 500000 \
--log_interval 100 \
--save_interval 500000 \
--num_episodes 1 \
--n_size 5 \
--n_exec_steps 1 \
--goal_padding False \
--predict_noise False \
--pad False

# ~fourth trajectory
python janner_diffuser.py \
--use_wandb True \
--name 'JannerInpaint-h128-clean' \
--env 'gridland-n5' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 128 \
--batch_size 128 \
--gradient_steps 500000 \
--log_interval 100 \
--save_interval 500000 \
--num_episodes 1 \
--n_size 5 \
--n_exec_steps 1 \
--goal_padding False \
--predict_noise False \
--pad False

# ~eighth trajectory
python janner_diffuser.py \
--use_wandb True \
--name 'JannerInpaint-h64-Clean' \
--env 'gridland-n5' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 64 \
--batch_size 128 \
--gradient_steps 500000 \
--log_interval 100 \
--save_interval 500000 \
--num_episodes 1 \
--n_size 5 \
--n_exec_steps 1 \
--goal_padding False \
--predict_noise False \
--pad False

# ~smol trajectory
python janner_diffuser.py \
--use_wandb True \
--name 'JannerInpaint-h32-Clean' \
--env 'gridland-n5' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 32 \
--batch_size 128 \
--gradient_steps 500000 \
--log_interval 100 \
--save_interval 500000 \
--num_episodes 1 \
--n_size 5 \
--n_exec_steps 1 \
--goal_padding False \
--predict_noise False \
--pad False


#### below - predicting noise 
# full trajectory training
python janner_diffuser.py \
--use_wandb True \
--name 'JannerInpaint-hFull-Noise' \
--env 'gridland-n5' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 500 \
--batch_size 128 \
--gradient_steps 500000 \
--log_interval 100 \
--save_interval 500000 \
--num_episodes 1 \
--n_size 5 \
--n_exec_steps 1 \
--goal_padding False \
--predict_noise True \
--pad True

# ~half trajectory
python janner_diffuser.py \
--use_wandb True \
--name 'JannerInpaint-h256-Noise' \
--env 'gridland-n5' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 256 \
--batch_size 128 \
--gradient_steps 500000 \
--log_interval 100 \
--save_interval 500000 \
--num_episodes 1 \
--n_size 5 \
--n_exec_steps 1 \
--goal_padding False \
--predict_noise True \
--pad False

# ~fourth trajectory
python janner_diffuser.py \
--use_wandb True \
--name 'JannerInpaint-h128-Noise' \
--env 'gridland-n5' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 128 \
--batch_size 128 \
--gradient_steps 500000 \
--log_interval 100 \
--save_interval 500000 \
--num_episodes 1 \
--n_size 5 \
--n_exec_steps 1 \
--goal_padding False \
--predict_noise True \
--pad False

# ~eighth trajectory
python janner_diffuser.py \
--use_wandb True \
--name 'JannerInpaint-h64-Noise' \
--env 'gridland-n5' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 64 \
--batch_size 128 \
--gradient_steps 500000 \
--log_interval 100 \
--save_interval 500000 \
--num_episodes 1 \
--n_size 5 \
--n_exec_steps 1 \
--goal_padding False \
--predict_noise True \
--pad False

# ~smol trajectory
python janner_diffuser.py \
--use_wandb True \
--name 'JannerInpaint-h32-Noise' \
--env 'gridland-n5' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 32 \
--batch_size 128 \
--gradient_steps 500000 \
--log_interval 100 \
--save_interval 500000 \
--num_episodes 1 \
--n_size 5 \
--n_exec_steps 1 \
--goal_padding False \
--predict_noise True \
--pad False