import os
import numpy as np
import traceback
import torch
import optuna

from argparse import ArgumentParser, Namespace
from datetime import datetime
from optuna.visualization import plot_optimization_history, plot_slice
from test import test_autoregressive, test_autoregressive_node_only
from torch.nn import MSELoss
from torch_geometric.loader import DataLoader
from training import NodeRegressionTrainer, DualRegressionTrainer, DualAutoRegressiveTrainer
from typing import List, Optional, Tuple
from utils import ValidationStats, Logger, file_utils
from utils.hp_search_utils import HYPERPARAMETER_CHOICES, load_datasets, load_model,\
    create_cross_val_dataset_files, create_temp_dirs, delete_temp_dirs, get_static_config

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--hyperparameters", type=str, choices=HYPERPARAMETER_CHOICES, nargs='+', required=True, help='Hyperparameters to search for')
    parser.add_argument("--config", type=str, required=True, help='Path to training config file')
    parser.add_argument("--model", type=str, required=True, help='Model to use for training')
    parser.add_argument("--summary_file", type=str, required=True, help='Dataset summary file for hyperparameter search. Events in file will be used for cross-validation')
    parser.add_argument("--num_trials", type=int, default=20, help='Number of trials for hyperparameter search')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    return parser.parse_args()

def get_hyperparam_search_config() -> Tuple[bool, bool, bool]:
    hyperparams_to_search = args.hyperparameters

    use_global_mass_loss = 'global_mass_loss' in hyperparams_to_search
    use_local_mass_loss = 'local_mass_loss' in hyperparams_to_search
    use_edge_pred_loss = 'edge_pred_loss' in hyperparams_to_search
    return use_global_mass_loss, use_local_mass_loss, use_edge_pred_loss

def save_cross_val_results(trainer, validation_stats: ValidationStats, run_id: str, model_postfix: str):
    curr_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = f'{args.model}_{curr_date_str}{model_postfix}'
    stats_dir = train_config['stats_dir']
    if stats_dir is not None:
        if not os.path.exists(stats_dir):
            os.makedirs(stats_dir)

        saved_metrics_path = os.path.join(stats_dir, f'{model_name}_train_stats.npz')
        trainer.save_stats(saved_metrics_path)

    output_dir = test_config['output_dir']
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get filename from model path
        saved_metrics_path = os.path.join(output_dir, f'{model_name}_runid_{run_id}_test_metrics.npz')
        validation_stats.save_stats(saved_metrics_path)

def cross_validate(global_mass_loss_percent: Optional[float],
                   local_mass_loss_percent: Optional[float],
                   edge_pred_loss_percent: Optional[float],
                   save_stats_for_first: bool = False) -> float | Tuple[float, float]:
    val_rmses = []
    if use_edge_pred_loss:
        val_edge_rmses = []
    for i, run_id in enumerate(hec_ras_run_ids):
        logger.log(f'Cross-validating with Run ID {run_id} as the test set...\n')

        storage_mode = config['dataset_parameters']['storage_mode']
        train_dataset, test_dataset = load_datasets(run_id,
                                                    base_dataset_config,
                                                    use_global_mass_loss,
                                                    use_local_mass_loss,
                                                    storage_mode)
        model = load_model(args.model, model_params, train_dataset, args.device)
        delta_t = train_dataset.timestep_interval

        # ============ Training Phase ============
        criterion = MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])

        if global_mass_loss_percent is None:
            global_mass_loss_percent = config['loss_func_parameters']['global_mass_loss_percent']
        if local_mass_loss_percent is None:
            local_mass_loss_percent = config['loss_func_parameters']['local_mass_loss_percent']
        if edge_pred_loss_percent is None:
            edge_pred_loss_percent = config['loss_func_parameters']['edge_pred_loss_percent']
        trainer_params = {
            'model': model,
            'dataset': train_dataset,
            'optimizer': optimizer,
            'loss_func': criterion,
            'use_global_loss': use_global_mass_loss,
            'global_mass_loss_percent': global_mass_loss_percent,
            'use_local_loss': use_local_mass_loss,
            'local_mass_loss_percent': local_mass_loss_percent,
            'delta_t': delta_t,
            'batch_size': train_config['batch_size'],
            'num_epochs': train_config['num_epochs'],
            'logger': logger,
            'device': args.device,
        }
        if use_edge_pred_loss:
            if train_config.get('autoregressive', False):
                num_timesteps = train_config['autoregressive_timesteps']
                curriculum_epochs = train_config['curriculum_epochs']
                logger.log(f'Using autoregressive training with intervals of {num_timesteps} timessteps and curriculum learning for {curriculum_epochs} epochs')

                trainer = DualAutoRegressiveTrainer(**trainer_params, edge_pred_loss_percent=edge_pred_loss_percent, num_timesteps=num_timesteps, curriculum_epochs=curriculum_epochs)
            else:
                trainer = DualRegressionTrainer(**trainer_params, edge_pred_loss_percent=edge_pred_loss_percent)
        else:
            trainer = NodeRegressionTrainer(**trainer_params)

        trainer.training_stats.log = lambda x : None  # Suppress training stats logging to console
        trainer.train()
        trainer.training_stats.log = logger.log  # Restore logging to console

        trainer.print_stats_summary()

        # ============ Testing Phase ============
        logger.log('\nValidating model...')

        rollout_start = test_config['rollout_start']
        rollout_timesteps = test_config['rollout_timesteps']
        validation_stats = ValidationStats(logger=logger)
        if use_edge_pred_loss:
            test_autoregressive(model, test_dataset, 0, validation_stats, rollout_start, rollout_timesteps, args.device, include_physics_loss=False)
        else:
            test_autoregressive_node_only(model, test_dataset, 0, validation_stats, rollout_start, rollout_timesteps, args.device, include_physics_loss=False)

        avg_rmse = validation_stats.get_avg_rmse()
        val_rmses.append(avg_rmse)
        logger.log(f'Event {run_id} RMSE: {avg_rmse:.4e}')

        if use_edge_pred_loss:
            avg_edge_rmse = validation_stats.get_avg_edge_rmse()
            val_edge_rmses.append(avg_edge_rmse)
            logger.log(f'Event {run_id} Edge RMSE: {avg_edge_rmse:.4e}')

        # ============ Saving stats (optional) ============
        if save_stats_for_first and i == 0:
            model_postfix = ''
            if use_global_mass_loss:
                model_postfix += f'_g{global_mass_loss_percent}'
            if use_local_mass_loss:
                model_postfix += f'_l{local_mass_loss_percent}'
            if use_edge_pred_loss:
                model_postfix += f'_e{edge_pred_loss_percent}'

            save_cross_val_results(trainer, validation_stats, run_id, model_postfix)

    def get_avg_rmse(rmses: List[float]) -> float:
        np_rmses = np.array(rmses)
        is_finite = np.isfinite(np_rmses)
        if np.any(is_finite):
            return np_rmses[is_finite].mean()
        return 1e10

    avg_val_rmse = get_avg_rmse(val_rmses)
    logger.log(f'\nAverage RMSE across all events: {avg_val_rmse:.4e}')
    if not use_edge_pred_loss:
        return avg_val_rmse

    avg_val_edge_rmse = get_avg_rmse(val_edge_rmses)
    logger.log(f'Average Edge RMSE across all events: {avg_val_edge_rmse:.4e}')
    return avg_val_rmse, avg_val_edge_rmse

def objective(trial: optuna.Trial) -> float:
    global_mass_loss_percent = trial.suggest_float('global_mass_loss_percent', 0.001, 0.5, log=True) if use_global_mass_loss else None
    local_mass_loss_percent = trial.suggest_float('local_mass_loss_percent', 0.001, 0.5, log=True) if use_local_mass_loss else None
    edge_pred_loss_percent = trial.suggest_float('edge_pred_loss_percent', 0.25, 0.35, step=0.05) if use_edge_pred_loss else None

    logger.log(f'Hyperparameters: global_mass_loss_percent={global_mass_loss_percent}, local_mass_loss_percent={local_mass_loss_percent}, edge_pred_loss_percent={edge_pred_loss_percent}')

    return cross_validate(global_mass_loss_percent,
                          local_mass_loss_percent,
                          edge_pred_loss_percent,
                          save_stats_for_first=True)

def plot_hyperparameter_search_results(study: optuna.Study):
    stats_dir = train_config['stats_dir']
    if stats_dir is None:
        return

    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    if use_edge_pred_loss:
        fig = plot_optimization_history(study, target=lambda t: t.values[0], target_name='Node RMSE')
        fig.write_html(os.path.join(stats_dir, 'optimization_history.html'))

        fig = plot_optimization_history(study, target=lambda t: t.values[1], target_name='Edge RMSE')
        fig.write_html(os.path.join(stats_dir, 'edge_optimization_history.html'))

        fig = plot_slice(study, target=lambda t: t.values[0], target_name='Node RMSE')
        fig.write_html(os.path.join(stats_dir, 'slice_plot.html'))

        fig = plot_slice(study, target=lambda t: t.values[1], target_name='Edge RMSE')
        fig.write_html(os.path.join(stats_dir, 'edge_slice_plot.html'))
    else:
        fig = plot_optimization_history(study, target_name='RMSE')
        fig.write_html(os.path.join(stats_dir, 'optimization_history.html'))

        fig = plot_slice(study, target_name='RMSE')
        fig.write_html(os.path.join(stats_dir, 'slice_plot.html'))

if __name__ == '__main__':
    args = parse_args()
    config = file_utils.read_yaml_file(args.config)

    # Initialize logger
    train_config = config['training_parameters']
    log_path = train_config['log_path']
    logger = Logger(log_path=log_path)

    try:
        logger.log('================================================')

        use_global_mass_loss, use_local_mass_loss, use_edge_pred_loss = get_hyperparam_search_config()
        assert use_global_mass_loss or use_local_mass_loss or use_edge_pred_loss, 'At least one hyperparameter must be selected for search'
        assert not use_edge_pred_loss or 'NodeEdgeGNN' in args.model, 'Edge prediction loss can only be used with NodeEdgeGNN model'

        if args.seed is not None:
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        current_device = torch.cuda.get_device_name(args.device) if args.device != 'cpu' else 'CPU'
        logger.log(f'Using device: {current_device}')

        static_config = get_static_config(config, args.model, logger)
        base_dataset_config, model_params, train_config, test_config = static_config

        # Begin hyperparameter search
        root_dir = base_dataset_config['root_dir']
        raw_temp_dir_path, processed_temp_dir_path = create_temp_dirs(root_dir)
        hec_ras_run_ids = create_cross_val_dataset_files(root_dir, args.summary_file)

        study_kwargs = {}
        if use_edge_pred_loss:
            study_kwargs['directions'] = ['minimize', 'minimize']
        else:
            study_kwargs['direction'] = 'minimize'

        study = optuna.create_study(**study_kwargs)
        logger.log(f'Using sampler: {study.sampler.__class__.__name__ if study.sampler else None}')
        logger.log(f'Using pruner: {study.pruner.__class__.__name__ if study.pruner else None}')
        logger.log(f'Running hyperparameter search for {args.num_trials} trials...')
        study.optimize(objective, n_trials=args.num_trials)

        if use_edge_pred_loss:
            logger.log('Best hyperparameters found:')
            for trial in study.best_trials:
                logger.log(f'Trial {trial.number}:')
                for key, value in trial.params.items():
                    logger.log(f'\t{key}: {value}')
                objective_values_str = ', '.join([f'{v:.4e}' for v in trial.values])
                logger.log(f'\tObjective values: {objective_values_str}')
        else:
            logger.log('Best hyperparameters found:')
            for key, value in study.best_params.items():
                logger.log(f'{key}: {value}')
            logger.log(f'Best objective value: {study.best_value:.4e}')

        # Plot hyperparameter search results
        plot_hyperparameter_search_results(study)

        # Clean up temporary directories
        delete_temp_dirs(raw_temp_dir_path, processed_temp_dir_path)

        logger.log('================================================')

    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')
