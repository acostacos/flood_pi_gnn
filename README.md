# Physics-informed Graph Neural Networks for Operational Flood Modelling

## Setup

### Environment
1. Create a virtual environment (with either conda or venv).
```bash
python -m venv venv

source venv/bin/activate # Linux
venv/Scripts/Activate.ps1 # Windows
```
2. Install PyTorch based on your CUDA version. Currently, the repository uses PyTorch version 2.5.1. Replace `${CUDA}` with the apporpriate CUDA version for your machine (ex. `${CUDA}` -> 124 for CUDA 12.4).
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu${CUDA}
```

3. Install PyTorch Geometric based on your PyTorch and CUDA version. Again, replace `${CUDA}` with the apporpriate CUDA version for your machine.

```bash
# Main library
pip install torch_geometric

# Additional libraries
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.1+cu${CUDA}.html
```

4. Install the remaining dependencies.

```bash
pip install -r requirements.txt
```

### Data

1. Request for access to the data.
2. Place all of necessary files in the `data/datasets/raw` folder. This is the default location but you may use other paths which must be defined in your `config.yaml` file.
3. Important files to have for dataset:
  - Node shape file (.shp)
  - Links shape file (.shp)
  - DEM file (.tif)
  - HEC-RAS simulation files (.hdf)
  - Summary file for training events (.csv)
  - Summary file for testing events (.csv)

For more information, see the dataset documentation.

## Files Overview

Below are the list of entry points for the application that you may run.

| File | Description | Arguments |
|---|---|---|
| `train.py` | Train the model with the parameters specified in the config file. | `--config`, `--model`, `--with_test` `--seed` `--device` `--debug` |
| `test.py` | Perform inference using the specified model checkpoint with test data. | `--config`, `--model`, `--model_path`, `--seed`, `--device`, `--debug` |
| `hp_search.py` | Perform a Bayesian hyperparameter search with the specified hyperparameters and events. | `--config`, `--hparam_config`, `--model`, `--seed`, `--device` |
| `eda.ipynb` | Jupyter notebook that gives an overview and analysis of the data. | N/A |
| `view_results.ipynb` | Jupyter notebook where you may view the results of model training and testing. | N/A |

Notes
- .sh files are mainly used for running programs in the slurm cluster.

## Code Structure

The code is categorized in different folder based on their specific purpose. Below is an overview of all the folders.

| Folder | Description |
|---|---|
| [configs](https://github.com/acostacos/flood_pi_gnn/tree/master/configs) | Contains all the config files used to specify training and testing parameters. |
| [constants](https://github.com/acostacos/flood_pi_gnn/tree/master/constants) | Contains constants used throughout the codebase. |
| [data](https://github.com/acostacos/flood_pi_gnn/tree/master/data) | Contains the raw data and Dataset classes for accessing this data. |
| [loss](https://github.com/acostacos/flood_pi_gnn/tree/master/loss) | Contains custom loss functions used for training (ex. physics-informed loss). |
| [models](https://github.com/acostacos/flood_pi_gnn/tree/master/models) | Contains different GNN model architectures. |
| [testing](https://github.com/acostacos/flood_pi_gnn/tree/master/testing) | Contains Tester classes used to test the model. |
| [training](https://github.com/acostacos/flood_pi_gnn/tree/master/training) | Contains Trainer classes used to train the model. |
| [utils](https://github.com/acostacos/flood_pi_gnn/tree/master/utils) | Contains various utility classes and objects. |
