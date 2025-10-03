#!/bin/sh
#SBATCH --job-name=test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=32000

. venv/bin/activate

srun python test.py --config 'configs/global_local_weight_search/global_0.1_and_local_0.1_config.yaml' --model 'NodeEdgeGNN' --model_path ''
srun python test.py --config 'configs/global_local_weight_search/global_0.1_and_local_0.01_config.yaml' --model 'NodeEdgeGNN' --model_path ''
srun python test.py --config 'configs/global_local_weight_search/global_0.1_and_local_0.001_config.yaml' --model 'NodeEdgeGNN' --model_path ''
srun python test.py --config 'configs/global_local_weight_search/global_0.1_and_local_0.0001_config.yaml' --model 'NodeEdgeGNN' --model_path ''

# srun python test.py --config 'configs/config.yaml' --model 'GAT' --model_path ''
# srun python test.py --config 'configs/config.yaml' --model 'GCN' --model_path ''
