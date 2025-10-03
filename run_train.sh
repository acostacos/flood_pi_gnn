#!/bin/sh
#SBATCH --job-name=train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=64000
#SBATCH --time=2880

. venv/bin/activate

srun python train.py --config 'configs/global_local_weight_search/global_0.0001_and_local_0.1_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/global_local_weight_search/global_0.0001_and_local_0.01_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/global_local_weight_search/global_0.0001_and_local_0.001_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/global_local_weight_search/global_0.0001_and_local_0.0001_config.yaml' --model 'NodeEdgeGNN'

# srun python train.py --config 'configs/config.yaml' --model 'GAT'
# srun python train.py --config 'configs/config.yaml' --model 'GCN'
