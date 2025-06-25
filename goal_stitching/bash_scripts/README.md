## Notes on Batch Job Organization

The way I have things set up is that each algorithm (Janner Diffuser, Decision Diffuser, etc.) has a slurm-job script wrapper for it. 

This wrapper just submits xyz algorithm as a job to the SLURM scheduler, directly passing any arguments you give it to the Python script.

These wrappers are then called by scripts for specific experiments, which submits each job to the SLURM scheduler sequentially. Note that this does not mean they are executed sequentially - each submission takes a fraction of a second, and the scheduler will decide when to run each individual one. 

I think this is a relatively drama-free and intuitive way of doing this.

#### Wrapper Notes
Each batch job needs to do some setup before running the Python script for the algorithm, notably, activating the correct Conda environment and setting up WanDB logging. This requires your individual WanDB API key to be used. If you don't want to do any logging you can just set the WanDB argument to False in the arguments passed to SLURM wrapper. 