import numpy as np
import os
import traceback
import torch
import gc
import random

from argparse import ArgumentParser, Namespace
from datetime import datetime
from data import dataset_factory, FloodEventDataset
from models import model_factory
from test import get_test_dataset_config, run_test
from training import trainer_factory
from typing import Dict, Optional, Tuple
from utils import Logger, file_utils, train_utils

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config", type=str, required=True, help='Path to training config file')
    parser.add_argument("--model", type=str, required=True, help='Model to use for training')
    parser.add_argument("--with_test", type=bool, default=False, help='Whether to run test after training')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    parser.add_argument("--debug", type=bool, default=False, help='Add debug messages to output')
    return parser.parse_args()

def load_dataset(config: Dict, args: Namespace, logger: Logger) -> Tuple[FloodEventDataset, Optional[FloodEventDataset]]:
    dataset_parameters = config['dataset_parameters']
    root_dir = dataset_parameters['root_dir']
    train_dataset_parameters = dataset_parameters['training']
    loss_func_parameters = config['loss_func_parameters']
    base_datset_config = {
        'root_dir': root_dir,
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
        'with_global_mass_loss': loss_func_parameters['use_global_mass_loss'],
        'with_local_mass_loss': loss_func_parameters['use_local_mass_loss'],
        'debug': args.debug,
        'logger': logger,
        'force_reload': True,
    }

    dataset_summary_file = train_dataset_parameters['dataset_summary_file']
    event_stats_file = train_dataset_parameters['event_stats_file']
    storage_mode = dataset_parameters['storage_mode']

    train_config = config['training_parameters']
    early_stopping_patience = train_config['early_stopping_patience']
    if early_stopping_patience is None:
        # No validation dataset needed
        dataset_config = {
            'mode': 'train',
            'dataset_summary_file': dataset_summary_file,
            'event_stats_file': event_stats_file,
            **base_datset_config,
        }
        logger.log(f'Using dataset configuration: {dataset_config}')

        dataset = dataset_factory(storage_mode, autoregressive=False, **dataset_config)
        logger.log(f'Loaded train dataset with {len(dataset)} samples')
        return dataset, None

    percent_validation = train_config['val_split_percent']
    assert percent_validation is not None, 'Validation split percentage must be specified if early stopping is used.'

    # Split dataset into training and validation sets for autoregressive training
    logger.log(f'Splitting dataset into training and validation sets with {percent_validation * 100}% for validation')
    train_summary_file, val_summary_file = train_utils.split_dataset_events(root_dir, dataset_summary_file, percent_validation)

    autoregressive_train_params = train_config['autoregressive']
    autoregressive_enabled = autoregressive_train_params.get('enabled', False)
    train_event_stats_file = train_summary_file.replace(dataset_summary_file, event_stats_file)
    train_dataset_config = {
        'mode': 'train',
        'dataset_summary_file': train_summary_file,
        'event_stats_file': train_event_stats_file,
        **base_datset_config,
    }
    if autoregressive_enabled:
        train_dataset_config.update({
            'num_label_timesteps': autoregressive_train_params['total_num_timesteps'],
        })
    logger.log(f'Using training dataset configuration: {train_dataset_config}')
    train_dataset = dataset_factory(storage_mode, autoregressive=autoregressive_enabled, **train_dataset_config)

    val_event_stats_file = val_summary_file.replace(dataset_summary_file, event_stats_file)
    val_dataset_config = {
        'mode': 'test',
        'dataset_summary_file': val_summary_file,
        'event_stats_file': val_event_stats_file,
        **base_datset_config,
    }
    logger.log(f'Using validation dataset configuration: {val_dataset_config}')
    val_dataset = dataset_factory(storage_mode, autoregressive=False, **val_dataset_config)

    logger.log(f'Loaded train dataset with {len(train_dataset)} samples and validation dataset with {len(val_dataset)} samples')
    return train_dataset, val_dataset

def run_train(model: torch.nn.Module,
              model_name: str,
              train_dataset: FloodEventDataset,
              logger: Logger,
              config: Dict,
              val_dataset: Optional[FloodEventDataset] = None,
              stats_dir: Optional[str] = None,
              model_dir: Optional[str] = None,
              device: str = 'cpu') -> str:
        train_config = config['training_parameters']

        # Loss function and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['adam_weight_decay'])
        logger.log(f'Using Adam optimizer with learning rate {train_config["learning_rate"]} and weight decay {train_config["adam_weight_decay"]}')

        base_trainer_params = train_utils.get_trainer_config(model_name, config, logger)
        trainer_params = {
            'model': model,
            'dataset': train_dataset,
            'val_dataset': val_dataset,
            'optimizer': optimizer,
            'logger': logger,
            'device': device,
            **base_trainer_params,
        }

        autoregressive_train_config = train_config['autoregressive']
        autoregressive_enabled = autoregressive_train_config.get('enabled', False)
        trainer = trainer_factory(model_name, autoregressive_enabled, **trainer_params)
        trainer.train()

        trainer.print_stats_summary()

        # Save training stats and model
        curr_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if stats_dir is not None:
            if not os.path.exists(stats_dir):
                os.makedirs(stats_dir)

            saved_metrics_path = os.path.join(stats_dir, f'{model_name}_{curr_date_str}_train_stats.npz')
            trainer.save_stats(saved_metrics_path)

        model_path = f'{model_name}_{curr_date_str}.pt'
        if model_dir is not None:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_path = os.path.join(model_dir, f'{model_name}_{curr_date_str}.pt')
            trainer.save_model(model_path)

        return model_path

def main():
    args = parse_args()
    config = file_utils.read_yaml_file(args.config)

    train_config = config['training_parameters']
    log_path = train_config['log_path']
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
        train_dataset, val_dataset = load_dataset(config, args, logger)

        # Model
        model_params = config['model_parameters'][args.model]
        base_model_params = {
            'static_node_features': train_dataset.num_static_node_features,
            'dynamic_node_features': train_dataset.num_dynamic_node_features,
            'static_edge_features': train_dataset.num_static_edge_features,
            'dynamic_edge_features': train_dataset.num_dynamic_edge_features,
            'previous_timesteps': train_dataset.previous_timesteps,
            'device': args.device,
        }
        model_config = {**model_params, **base_model_params}
        model = model_factory(args.model, **model_config)
        logger.log(f'Using model: {args.model}')
        logger.log(f'Using model configuration: {model_config}')
        num_train_params = model.get_model_size()
        logger.log(f'Number of trainable model parameters: {num_train_params}')

        checkpoint_path = train_config.get('checkpoint_path', None)
        if checkpoint_path is not None:
            logger.log(f'Loading model from checkpoint: {checkpoint_path}')
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=args.device))

        stats_dir = train_config['stats_dir']
        model_dir = train_config['model_dir']
        model_path = run_train(model=model,
                               model_name=args.model,
                               train_dataset=train_dataset,
                               val_dataset=val_dataset,
                               logger=logger,
                               config=config,
                               stats_dir=stats_dir,
                               model_dir=model_dir,
                               device=args.device)

        logger.log('================================================')

        if not args.with_test:
            return

        # =================== Testing ===================
        logger.log(f'Starting testing for model: {model_path}')

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
            'debug': args.debug,
            'logger': logger,
            'force_reload': True,
        }
        test_dataset_config = get_test_dataset_config(base_datset_config, config)
        logger.log(f'Using test dataset configuration: {test_dataset_config}')

        # Clear memory before loading test dataset
        del dataset
        gc.collect()

        storage_mode = dataset_parameters['storage_mode']
        dataset = dataset_factory(storage_mode, autoregressive=False, **test_dataset_config)
        logger.log(f'Loaded test dataset with {len(dataset)} samples')

        logger.log(f'Using model checkpoint for {args.model}: {model_path}')
        logger.log(f'Using model configuration: {model_config}')

        test_config = config['testing_parameters']
        rollout_start = test_config['rollout_start']
        rollout_timesteps = test_config['rollout_timesteps']
        output_dir = test_config['output_dir']
        run_test(model=model,
                 model_path=model_path,
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
