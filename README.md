# Shuffle Variations

![GitHub stars](https://github.com/BraSDon/paper)
![GitHub forks](https://github.com/BraSDon/paper)
![GitHub license](https://github.com/BraSDon/paper)

Repository for experimenting with different inter-epoch shuffling methods in data parallel training of neural networks.

## Introduction

This repository is dedicated to the exploration of various techniques for shuffling data between epochs during the training of neural networks. The primary goals of this research project are:

1. **Reducing Communication Overhead:** Find methods that require less communication between nodes/ranks in distributed training environments.
2. **Sequential Data Access:** Develop strategies that allow for sequential data access during training.
3. **Maintaining Quality Metrics:** Ensure that the quality of models trained using these shuffling methods remains state-of-the-art (SOTA).

## Key Objectives

- Identify efficient data shuffling techniques for neural network training.
- Investigate ways to minimize communication overhead in distributed training setups.
- Preserve or improve the quality of trained models while optimizing data shuffling.

## Who Is This For?

This repository is relevant for you if:

- You're interested in speeding up the training of neural networks by reducing shuffling overhead.
- You're exploring the usage of Data Parallelism (DDP) on a Slurm cluster.
- You're curious about innovative techniques for managing data in deep learning projects.

## Getting Started
1. Clone the repository
2. Install requirements.txt
3. Adjust the system-config.yaml file to your needs
4. Add your own run-config in the run-configs folder

Slurm Cluster:
5. Add your own Slurm batch script in the slurm folder
6. Submit the Slurm job

Local (4 processes fixed!, CPU execution):
5. torchrun --nproc_per_node=4 main.py --config_path=run-configs/<your-config>.yaml

## Repository Structure

The repository is organized as follows:

```markdown
.
├── README.md
├── data
├── notebooks
├── run-configs
├── slurm
├── src
│   ├── main.py
│   ├── data
│   │   ├── data.py
│   │   ├── datasets.py
│   │   ├── partition.py
│   │   ├── sorted_dataset.py
│   ├── models
│   │   ├── models.py
│   ├── training
│   │   ├── train.py
│   │   ├── custom_sampler.py
│   │   ├── stratified_sampler.py
│   ├── util
│   │   ├── cases.py
│   │   ├── helper.py
│   ├── visualization
├── test
├── wandb
├── system-config.yaml

```

## system-config.yaml
This file contains all configuration elements that are system specific and run-independent.
This includes the ddp port, the system type (server | local) and the entire specification of available datasets.
Therefore, only this file needs to be changed when adding/removing/updating datasets.

The file is structured as follows:
```
system: (server | local)
ddp:
  port: <port>
datasets:
  <dataset_name>
    path: <path_to_dataset>
    load-function:
      module: <module_name>
      type: (generic | built-in)
      name: <function_name>
    transforms:
      train:
        - name: <transform_name>
          kwargs:
            <param_name>: <param_value>
        ...
      test:
        ...
```
The load-function specifies how a dataset can be loaded 
(e.g. torchvision.datasets.cifar.CIFAR10 or torchvision.datasets.ImageFolder). 
Where the type specifies which arguments are passed to the function 
(train=(True | False) is set only when using built-in datasets).
The transforms are applied to the dataset in the given order.

## Potential Pitfalls/Errors
- When using torch.hub.load() the repo might get downloaded but not unzipped. 
  This will result in a FileNotFoundError of the hubconf.py file. 
  To fix this, manually unzip the downloaded file and make sure it has the correct name (e.g. pytorch_vision_v0.10.0).
- To efficiently sort the dataset and calculate the label frequencies, the script requires the dataset object to have 
  an attribute called "targets". This should contain the labels of the dataset (as integers) 
  in the same order as the dataset itself.