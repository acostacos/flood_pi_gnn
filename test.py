import numpy as np
import traceback
import torch
import os
import random

from argparse import ArgumentParser, Namespace
from constants import EDGE_MODELS
from data import dataset_factory, FloodEventDataset
from models import model_factory
from testing import DualAutoregressiveTester, EdgeAutoregressiveTester, NodeAutoregressiveTester
from typing import Dict, Optional
from utils import Logger, file_utils

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config", type=str, required=True, help='Path to training config file')
    parser.add_argument("--model", type=str, required=True, help='Model to use for validation')
    parser.add_argument('--model_path', required=True, default=None, help='Path to trained model file')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    return parser.parse_args()

def get_test_dataset_config(base_datset_params: Dict, config: Dict) -> Dict:
    dataset_parameters = config['dataset_parameters']
    test_dataset_parameters = dataset_parameters['testing']
    test_dataset_config = {
        **base_datset_params,
        'mode': 'test',
        'dataset_summary_file': test_dataset_parameters['dataset_summary_file'],
        'event_stats_file': test_dataset_parameters['event_stats_file'],
        'with_global_mass_loss': True,
        'with_local_mass_loss': True,
    }
    return test_dataset_config

def run_test(model: torch.nn.Module,
             model_path: str,
             dataset: FloodEventDataset,
             logger: Logger,
             rollout_start: int = 0,
             rollout_timesteps: Optional[int] = None,
             output_dir: Optional[str] = None,
             device: str = 'cpu'):
    log_test_config = {'rollout_start': rollout_start, 'rollout_timesteps': rollout_timesteps}
    logger.log(f'Using testing configuration: {log_test_config}')

    tester_params = {
        'model': model,
        'dataset': dataset,
        'rollout_start': rollout_start,
        'rollout_timesteps': rollout_timesteps,
        'include_physics_loss': True,
        'logger': logger,
        'device': device,
    }

    is_dual_model = 'NodeEdgeGNN' in model.__class__.__name__
    if is_dual_model:
        tester = DualAutoregressiveTester(**tester_params)
    elif model.__class__.__name__ in EDGE_MODELS:
        tester = EdgeAutoregressiveTester(**tester_params)
    else:
        tester = NodeAutoregressiveTester(**tester_params)
    tester.test()

    if output_dir is not None:
        # Save model filename without extension
        model_filename = os.path.splitext(os.path.basename(model_path))[0] # Remove file extension
        tester.save_stats(output_dir, stats_filename_prefix=model_filename)

def main():
    args = parse_args()
    config = file_utils.read_yaml_file(args.config)

    test_config = config['testing_parameters']
    log_path = test_config['log_path']
    logger = Logger(log_path=log_path)

    try:
        logger.log('================================================')

        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            logger.log(f'Setting random seed to {args.seed}')

            # Enable deterministic behavior for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            logger.log('Enabled deterministic mode for reproducibility')

        current_device = torch.cuda.get_device_name(args.device) if args.device != 'cpu' else 'CPU'
        logger.log(f'Using device: {current_device}')

        # Dataset
        dataset_parameters = config['dataset_parameters']
        base_datset_config = {
            'root_dir': dataset_parameters['root_dir'],
            'nodes_shp_file': dataset_parameters['nodes_shp_file'],
            'edges_shp_file': dataset_parameters['edges_shp_file'],
            'features_stats_file': dataset_parameters['features_stats_file'],
            'previous_timesteps': dataset_parameters['previous_timesteps'],
            'normalize': dataset_parameters['normalize'],
            'timestep_interval': dataset_parameters['timestep_interval'],
            'spin_up_time': dataset_parameters['spin_up_time'],
            'time_from_peak': dataset_parameters['time_from_peak'],
            'inflow_boundary_nodes': dataset_parameters['inflow_boundary_nodes'],
            'outflow_boundary_nodes': dataset_parameters['outflow_boundary_nodes'],
        }
        base_datset_config = get_test_dataset_config(base_datset_config, config)
        logger.log(f'Using dataset configuration: {base_datset_config}')
        dataset_config = {
            **base_datset_config,
            'debug': args.debug,
            'logger': logger,
            'force_reload': True,
        }

        storage_mode = dataset_parameters['storage_mode']
        dataset = dataset_factory(storage_mode=storage_mode, autoregressive=False, **dataset_config)
        logger.log(f'Loaded dataset with {len(dataset)} samples')

        # Load model
        model_params = config['model_parameters'][args.model]
        previous_timesteps = dataset.previous_timesteps
        base_model_params = {
            'static_node_features': dataset.num_static_node_features,
            'dynamic_node_features': dataset.num_dynamic_node_features,
            'static_edge_features': dataset.num_static_edge_features,
            'dynamic_edge_features': dataset.num_dynamic_edge_features,
            'previous_timesteps': previous_timesteps,
            'device': args.device,
        }
        model_config = {**model_params, **base_model_params}
        model = model_factory(args.model, **model_config)
        model.load_state_dict(torch.load(args.model_path, weights_only=True))
        logger.log(f'Using model checkpoint for {args.model}: {args.model_path}')
        logger.log(f'Using model configuration: {model_config}')

        # Testing
        rollout_start = test_config['rollout_start']
        rollout_timesteps = test_config['rollout_timesteps']
        output_dir = test_config['output_dir']
        run_test(model=model,
                 model_path=args.model_path,
                 dataset=dataset,
                 logger=logger,
                 rollout_start=rollout_start,
                 rollout_timesteps=rollout_timesteps,
                 output_dir=output_dir,
                 device=args.device)

        logger.log('================================================')
    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')


if __name__ == '__main__':
    main()
