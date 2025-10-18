import os
import pandas as pd

from constants import EDGE_MODELS
from typing import Tuple

from .logger import Logger
from .file_utils import create_temp_dirs
from .model_utils import get_loss_func

def split_dataset_events(root_dir: str, dataset_summary_file: str, percent_validation: float) -> Tuple[str, str]:
    if not (0 < percent_validation < 1):
        raise ValueError(f'Invalid percent_split: {percent_validation}. Must be between 0 and 1.')

    raw_dir_path = os.path.join(root_dir, 'raw')
    dataset_summary_path = os.path.join(raw_dir_path, dataset_summary_file)

    assert os.path.exists(dataset_summary_path), f'Dataset summary file does not exist: {dataset_summary_path}'
    summary_df = pd.read_csv(dataset_summary_path)
    assert len(summary_df) > 0, f'No data found in summary file: {dataset_summary_path}'

    num_val_events = max(int(len(summary_df) * percent_validation), 1)
    split_idx = len(summary_df) - num_val_events

    TEMP_DIR_NAME = 'train_val_split'
    create_temp_dirs(raw_dir_path, folder_name=TEMP_DIR_NAME)

    train_rows = summary_df[:split_idx]
    train_df_file = os.path.join(TEMP_DIR_NAME, f'train_split_{dataset_summary_file}')
    train_rows.to_csv(os.path.join(raw_dir_path, train_df_file), index=False)

    val_rows = summary_df[split_idx:]
    val_df_file = os.path.join(TEMP_DIR_NAME, f'val_split_{dataset_summary_file}')
    val_rows.to_csv(os.path.join(raw_dir_path, val_df_file), index=False)

    return train_df_file, val_df_file

def get_trainer_config(model_name: str, config: dict, logger: Logger = None) -> dict:
    def log(msg):
        if logger:
            logger.log(msg)

    trainer_params = {}

    train_config = config['training_parameters']
    loss_func_parameters = config['loss_func_parameters']

    # Base Trainer parameters
    node_loss_func = loss_func_parameters['node_loss']
    edge_loss_func = loss_func_parameters['edge_loss']
    node_criterion = get_loss_func(node_loss_func, **loss_func_parameters.get(node_loss_func, {}))
    edge_criterion = get_loss_func(edge_loss_func, **loss_func_parameters.get(edge_loss_func, {}))
    loss_func = edge_criterion if model_name in EDGE_MODELS else node_criterion

    early_stopping_patience = train_config['early_stopping_patience']
    num_epochs = train_config['num_epochs']
    num_epochs_dyn_loss = train_config['num_epochs_dyn_loss']
    node_loss_weight = loss_func_parameters['node_loss_weight']
    log(f'Using dynamic loss weight adjustment for the first {num_epochs_dyn_loss}/{num_epochs} epochs')
    log(f'Applying importance weight of {node_loss_weight} to node prediction loss after scaling')
    base_config = {
        'num_epochs': num_epochs,
        'num_epochs_dyn_loss': num_epochs_dyn_loss,
        'batch_size': train_config['batch_size'],
        'gradient_clip_value': train_config['gradient_clip_value'],
        'loss_func': loss_func,
        'node_loss_weight': node_loss_weight,
        'early_stopping_patience': early_stopping_patience,
    }
    log(f'Using training configuration: {base_config}')
    trainer_params.update(base_config)

    # Physics-informed training parameters
    if model_name not in EDGE_MODELS:
        use_global_mass_loss = loss_func_parameters['use_global_mass_loss']
        global_mass_loss_scale = loss_func_parameters['global_mass_loss_scale']
        global_mass_loss_weight = loss_func_parameters['global_mass_loss_weight']
        if use_global_mass_loss:
            log(f'Using global mass conservation loss with scale {global_mass_loss_scale} with importance weight {global_mass_loss_weight}')

        use_local_mass_loss = loss_func_parameters['use_local_mass_loss']
        local_mass_loss_scale = loss_func_parameters['local_mass_loss_scale']
        local_mass_loss_weight = loss_func_parameters['local_mass_loss_weight']
        if use_local_mass_loss:
            log(f'Using local mass conservation loss with scale {local_mass_loss_scale} with importance weight {local_mass_loss_weight}')

        trainer_params.update({
            'use_global_loss': use_global_mass_loss,
            'global_mass_loss_scale': global_mass_loss_scale,
            'global_mass_loss_weight': global_mass_loss_weight,
            'use_local_loss': use_local_mass_loss,
            'local_mass_loss_scale': local_mass_loss_scale,
            'local_mass_loss_weight': local_mass_loss_weight,
        })

    # Autoregressive training parameters
    autoregressive_train_config = train_config['autoregressive']
    autoregressive_enabled = autoregressive_train_config.get('enabled', False)
    if autoregressive_enabled:
        init_num_timesteps = autoregressive_train_config['init_num_timesteps']
        total_num_timesteps = autoregressive_train_config['total_num_timesteps']
        learning_rate_decay = autoregressive_train_config['learning_rate_decay']
        max_curriculum_epochs = autoregressive_train_config['max_curriculum_epochs']
        log(f'Using autoregressive training for {init_num_timesteps}/{total_num_timesteps} timesteps and curriculum learning with patience {early_stopping_patience}, max {max_curriculum_epochs} epochs and learning rate decay {learning_rate_decay}')

        trainer_params.update({
            'init_num_timesteps': init_num_timesteps,
            'total_num_timesteps': total_num_timesteps,
            'learning_rate_decay': learning_rate_decay,
            'max_curriculum_epochs': max_curriculum_epochs,
        })

    # Node/Edge prediction parameters
    if 'NodeEdgeGNN' in model_name:
        edge_pred_loss_scale = loss_func_parameters['edge_pred_loss_scale']
        edge_pred_loss_weight = loss_func_parameters['edge_loss_weight']
        log(f'Using edge prediction loss with scale {edge_pred_loss_scale} with importance weight {edge_pred_loss_weight}')
        log(f"Using {edge_criterion.__class__.__name__} loss for edge prediction")
        trainer_params.update({
            'edge_loss_func': edge_criterion,
            'edge_pred_loss_scale': edge_pred_loss_scale,
            'edge_loss_weight': edge_pred_loss_weight,
        })

    return trainer_params

def divide_losses(losses: Tuple, divisor: float) -> Tuple:
    if divisor == 0:
        raise ValueError("Divisor cannot be zero")
    return tuple(loss / divisor for loss in losses)
