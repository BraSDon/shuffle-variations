# paper
Repository for experiments regarding &lt;paper>

## Dependencies

## Setup

## Usage

## Structure
```
.
├── README.md
├── notebooks
│    ├── README.md
├── src
    ├── README.md
    ├── main.py
    ├── data
    │    ├── README.md
    │    ├── data.py
    │    ├── transparent.py
    │    ├── stage
    │    │    ├── README.md
    │    │    ├── stage.py
    ├── models
    │    ├── README.md
    │    ├── models.py
    ├── training
    │    ├── README.md
    │    ├── train.py
    │    ├── evaluate.py
    │    ├── training_objects.py
    ├── util
    │    ├── README.md
    │    ├── parser.py
    │    ├── logger.py
    ├── visualization
    │    ├── README.md
    │    ├── visualize.py
├── test
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
  - name: <dataset_name>
    path: <path_to_dataset>
    load-function:
      module: <module_name>
      type: (generic | built-in)
      name: <function_name>
    transforms:
      - train:
        - name: <transform_name>
          kwargs:
            <param_name>: <param_value>
        ...
      - val:
        ...
      - test:
        ...
```
The load-function specifies how a dataset can be loaded 
(e.g. torchvision.datasets.cifar.CIFAR10 or torchvision.datasets.ImageFolder). 
Where the type specifies which arguments are passed to the function 
(train=(True | False) is set only when using built-in datasets).
The transforms are applied to the dataset in the given order.