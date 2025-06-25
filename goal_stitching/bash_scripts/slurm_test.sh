### just a script to test that submitting jobs and stuff works

# test janner diffuser run
sbatch janner_diffuser.slurm \
--use_wandb True \
--name 'JannerTest' \
--env_name 'scene-play-v0' \
--gradient_steps 100 \
--eval_every 50 \
--name 'Diffuser-Test' \
--checkpoints_path '../runs/' \
--invdyn_dir = '../common_models'

# test deciison diffuser run
sbatch decision_diffuser.slurm \
--use_wandb True \
--name 'DDTest' \
--env_name 'scene-play-v0' \
--gradient_steps 100 \
--eval_every 50 \
--name 'Diffuser-Test' \
--checkpoints_path '../runs/'
--invdyn_dir = '../common_models'