# NUNO
Non-Uniform Neural Operator

## Requirements
Python 3.10.8
- `matplotlib==3.6.2`
- `scikit-learn==1.2.0`
- `torch==1.13.0`
- `tqdm==4.64.1`
- `torch-geometric==2.2.0` (visit https://github.com/pyg-team/pytorch_geometric to install this package)
    - It is required by GraphNO. If you do no run this baseline, you can ignore this package.
- `sympy==1.11.1`
    - It is required by MWNO. If you do no run this baseline, you can ignore this package.

## Overview
This repository is organized as below:
- data (the folder for training and testing data)
    - channel ((2+1)D Channel Flow)
    - elasticity (2D Elasticity)
    - heatsink (3D Heatsink)
- src (the folder for scripts of running experiments)
    - channel ((2+1)D Channel Flow)
        Note: each Python file corresponds to a baseline.
        - `deeponet.py` (to run DeepONet)
        - ...
        - `ours_nuunet.py` (to run one of our methods, NU-U-Net)
    - elasticity (2D Elasticity)
        - ...
    - heatsink (3D Heatsink)
        - ...
- util (the folder for util functions)
- `README.md`

## Instruction

1. Install necessary dependencies. 
    For most experiments, you only need to install the first four libraries.
2. Download the dataset.
    For *2D Elasticity*, we refer to the repository https://github.com/neural-operator/Geo-FNO.

    Please download the dataset in the above repository.

    When you click the link, go into the folder `Geo-FNO/elasticity`. Download all the files in folder `Interp`, `Meshes`, `Omesh`, and `Rmesh`. Then put them all (directly move files not with original folders) into the folder `data/elasticity` of *this* repository.

    For *(2+1)D Channel Flow* and *3D Heatsink*, we refer to the link https://1drv.ms/u/s!ApNDtoKtGVC6k6AK3AYprkxiZW0qWA?e=7De5vc.

    Download the files in `NUNO/ChannelFlow` and `NUNO/Heatsink`, then put them into `data/channel` and `data/heatsink` of *this* repository, respectively.
3. Run the experiments.
    ```bash
    # In the root directory of this repository

    # To run scripts in 2D Elasticity
    python -m src.elasticity.ours_nufno
    # Or another baseline
    python -m src.elasticity.geofno

    # To run scripts in (2+1)D Channel Flow
    python -m src.channel.ours_nufno

    # To run scripts in 3D Heatsink
    python -m src.heatsink.ours_nufno
    ```
