### decagon stuff

python diffusion_planner.py \
--use_wandb True \
--guidance 'none' \
--name 'DP-UNet-Uncond-Full' \
--dataset_dir 'decagon_data/train_data.npy' \
--use_shift_equivariant_arch False \
--add_positional_encoding False \
--channel_size 64

python diffusion_planner.py \
--use_wandb True \
--guidance 'none' \
--name 'DP-UNet-Uncond-Medium' \
--dataset_dir 'decagon_data/train_data_medium.npy' \
--use_shift_equivariant_arch False \
--add_positional_encoding False \
--channel_size 64

python diffusion_planner.py \
--use_wandb True \
--guidance 'none' \
--name 'DP-UNet-Uncond-Small' \
--dataset_dir 'decagon_data/train_data_small.npy' \
--use_shift_equivariant_arch False \
--add_positional_encoding False \
--channel_size 64

python diffusion_planner.py \
--use_wandb True \
--guidance 'none' \
--name 'DP-UNet-Uncond-Tiny' \
--dataset_dir 'decagon_data/train_data_tiny.npy' \
--use_shift_equivariant_arch False \
--add_positional_encoding False \
--channel_size 64

python diffusion_planner.py \
--use_wandb True \
--guidance 'none' \
--name 'DP-UNet-Uncond-SM' \
--dataset_dir 'decagon_data/train_data_sm.npy' \
--use_shift_equivariant_arch False \
--add_positional_encoding False \
--channel_size 64


python diffusion_planner.py \
--use_wandb True \
--guidance 'none' \
--name 'DP-UNet-Uncond-SSM' \
--dataset_dir 'decagon_data/train_data_ssm.npy' \
--use_shift_equivariant_arch False \
--add_positional_encoding False \
--channel_size 64



python diffusion_planner.py \
--use_wandb True \
--guidance 'none' \
--name 'DP-UNet-Uncond-SMM' \
--dataset_dir 'decagon_data/train_data_smm.npy' \
--use_shift_equivariant_arch False \
--add_positional_encoding False \
--channel_size 64