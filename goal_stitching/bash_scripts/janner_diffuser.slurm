#!/bin/bash
### following are Vaughn parameters 
#SBATCH --job-name=janner_diffuser_run
#SBATCH --gres=gpu:t4:4
#SBATCH --qos=normal
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# activates Conda environment
conda activate gen_stitch

# activates WanDB with my (Quentin's) API key
python -m wandb.cli login 1225cab8d96c3720b5fbe61ca26431f08d95f5e9

# runs training
python ../janner_diffuser.py "$@"
