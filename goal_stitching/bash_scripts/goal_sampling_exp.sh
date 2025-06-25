### Training models for the experiments w/diffusion guidance


# Decision Diffuser
python decision_diffuser.py \
--use_wandb True \
--name 'CFGDiffuser-Geo-Start' \
--env 'cube-quadruple-play-v0' \
--checkpoints_path 'trained_models' \
--eval_interval 100000000 \
--invdyn_dir 'common_models' \
--horizon 512 \
--batch_size 128 \
--gradient_steps 500000 \
--log_interval 100 \
--save_interval 500000 \
--num_episodes 1 \
--label_dropout 0.25 \