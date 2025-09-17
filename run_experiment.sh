#!/bin/sh
#SBATCH --job-name=experiment
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=64000
#SBATCH --time=2880

. venv/bin/activate

# srun python train.py --config 'configs/autoregressive/2layer_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/autoregressive/4layer_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/autoregressive/6layer_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/autoregressive/8layer_config.yaml' --model 'NodeEdgeGNN'
# srun python train.py --config 'configs/autoregressive/10layer_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/autoregressive/12layer_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/autoregressive/14layer_config.yaml' --model 'NodeEdgeGNN'
srun python train.py --config 'configs/autoregressive/16layer_config.yaml' --model 'NodeEdgeGNN'

# srun python test.py --config 'configs/autoregressive/2layer_config.yaml' --model 'NodeEdgeGNN' --model_path 'num_gnn_layers/model/NodeEdgeGNN_2025-09-17_17-21-54_2layer.pt'
# srun python test.py --config 'configs/autoregressive/4layer_config.yaml' --model 'NodeEdgeGNN' --model_path 'num_gnn_layers/model/NodeEdgeGNN_2025-09-17_17-40-39_4layer.pt'
# srun python test.py --config 'configs/autoregressive/6layer_config.yaml' --model 'NodeEdgeGNN' --model_path 'num_gnn_layers/model/NodeEdgeGNN_2025-09-17_18-03-32_6layer.pt'
# srun python test.py --config 'configs/autoregressive/8layer_config.yaml' --model 'NodeEdgeGNN' --model_path 'num_gnn_layers/model/NodeEdgeGNN_2025-09-17_18-30-49_8layer.pt'
# srun python test.py --config 'configs/autoregressive/10layer_config.yaml' --model 'NodeEdgeGNN' --model_path 'num_gnn_layers/model/NodeEdgeGNN_2025-09-17_19-02-44_10layer.pt'
