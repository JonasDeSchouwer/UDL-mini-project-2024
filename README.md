# UDL-mini-project-2024

Welcome to the git repository for my UDL mini project! In this repository, I did the following three things:
1. I implemented the [VCL paper](https://arxiv.org/abs/1710.10628) from scratch in PyTorch.
2. I reproduced the Split MNIST experiment and compared the results when using an *Interleaved Training/Evaluation Scheme (ITES)* versus a *Sequential Training/Evaluation Scheme (STES)*.
3. I investigated the effects of applying LR decay to the shared weights of the multi-head neural network.


## Repository structure

### Python Files (code)

| **File** | **Description** |
|----------|-----------------|
| `datasets.py` | Code for loading the Split MNIST and Split FashionMNIST datasets. |
| `coreset.py` | Implementation of the RandomCoreset and KCenterCoreset algorithms. Further contains an implementation of the `MultiTaskDataContainer` that is used throughout this project.  |
| `models.py` | Implementation of the multi-head architecture, both as a standard and as a Bayesian neural network. |
| `utils.py` | Utility functions used throughout the project. |
| `experiments.py` | Main code for running the experiments and collecting results. |



### Notebooks (experiments)

| **File** | **Description** |
|----------|-----------------|
| `reproducing-base-results.ipynb` | Jupyter notebook where I reproduced the Split MNIST experiment and compared the results of ITES vs. STES. All this notebook does is define the right configurations for each experiment, call methods from `experiments.py` and print the results. |
| `lr-schedule.ipynb` | Jupyter notebook where I investigated the effects of applying LR decay to the shared weights. All this notebook does is define the right configurations for each experiment, call methods from `experiments.py` and print the results |
| `plot_figures.ipynb` | Jupyter notebook for plotting the figures used in the project report. |


### Folders (results)

| **Folder** | **Description** |
|----------|-----------------|
| `reproducing-base-results` | Contains the results of five experiments on the Split MNIST dataset with different training/evaluation schemes and coreset methods. Each experiment was run 10 times with different random seeds. |
| `lr-schedule` | Contains the results of six experiments on the FashionMNIST dataset with different LR decay rates for the shared weights. Each experiment was run 10 times with different random seeds. |
