#!/bin/sh
#SBATCH --job-name=experiment
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=64000
#SBATCH --time=2880

. venv/bin/activate

srun python train.py --config 'configs/autoregressive/2layer_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/autoregressive/4layer_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/autoregressive/6layer_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/autoregressive/8layer_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/autoregressive/10layer_config.yaml' --model 'NodeEdgeGNN'

# srun python test.py --config 'configs/autoregressive/2layer_config.yaml' --model 'NodeEdgeGNN' --model_path ''
# srun python test.py --config 'configs/autoregressive/4layer_config.yaml' --model 'NodeEdgeGNN' --model_path ''
# srun python test.py --config 'configs/autoregressive/6layer_config.yaml' --model 'NodeEdgeGNN' --model_path ''
# srun python test.py --config 'configs/autoregressive/8layer_config.yaml' --model 'NodeEdgeGNN' --model_path ''
# srun python test.py --config 'configs/autoregressive/10layer_config.yaml' --model 'NodeEdgeGNN' --model_path ''
