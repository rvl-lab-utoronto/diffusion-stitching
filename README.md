## What Do You Need for Diverse Trajectory Stitching in Diffusion Planning?

By Quentin Clark (contact: qtcc@cs.toronto.edu) and Florian Shkurti

Paper link: https://arxiv.org/abs/2505.18083

### Repository Layout

<ul>
  <li>goal_stitching: Contains most of the experiments in the paper, including the main results in the unconditional and goal-conditioned environments. </li>
  <li>data_scaling: Contains the data scaling experiments in the hexagon environment. </li>
  <li>language_cond: Contains incomplete, early experiments in a language-conditioned navigation task.</li>
  <li>optimality_stitching: Contains incomplete, early experiments with stitching from RvS-based algorithms like Decision Transformer in a different MuJoCo-based navigation environment. </li>
</ul> 

### Installation

We use Conda as our package mangager. Install conda, then run `conda env create -f environment.yml`. 

### Eq-Net

Our main software contribution is a new architecture for diffusion planning, which we call Eq-Net. We implemented Eq-Net using the CleanDiffuser [1]framework, which we copied into each repository for our experiments. We also put a copy of the architecture definition file in the main directory for convenience: `eqnet.py`. 

If you want to try Eq-Net in your own experiments, just topy `eqnet.py` into the following folder in CleanDiffuser: `\cleandiffuser\nn_diffusion`, then add `from .eqnet import EqNet` to `__init__.py` in that folder. Then can be used as part of a CleanDiffuser pipline like any other architecture.

### Recreating Experiments

#### Unconditional and Goal-Conditioned Experiments

To train the models required for our unconditional and goal-conditional experiments, run `bash paper_experiments.sh` in the goal_stitching folder. Generating results for individual trials can be found in the `goal_cond_results.ipynb` and `gridland_diversity_experiments.ipynb` notebooks. To visualize invididual generations, see the `generation_visualization_gridland_goal.ipynb` and `generation_visualization_gridland_uncond.ipynb` notebooks.

#### Data Scaling Experiments

To train the models for the data scaling experiments, run the `bash data_exp.sh` in the data_scaling folder. Results can be generated with the `data_scaling_results.ipynb` notebook. 

Due to space restrictions, we do not include the datasets for the data scaling experiments. To recreate that data set, run the `collect_decagon_data.ipynb` notebook. 

### Trained Models

Due to the large size of the trained models, we do not include them directly in the S.I. If you want access to the trained models, we can provide non-anonymous access (due to the large size, we have to host them on OneDrive where the username is accessible). To preserve anonymity for reviewers we will give the link by request. Text 425-802-4776 with the message "HELP - model link access" and we will send the shared link.


### References

[1] Dong, Zibin, et al. "Cleandiffuser: An easy-to-use modularized library for diffusion models in decision making." arXiv preprint arXiv:2406.09509 (2024).
