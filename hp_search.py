import os
import numpy as np
import traceback
import torch
import optuna
import random

from argparse import ArgumentParser, Namespace
from constants import EDGE_MODELS
from contextlib import redirect_stdout
from optuna.visualization import plot_optimization_history, plot_slice, plot_pareto_front
from pprint import pformat
from training import trainer_factory
from testing import DualAutoregressiveTester, NodeAutoregressiveTester, EdgeAutoregressiveTester
from typing import List, Tuple, Dict
from utils import Logger, file_utils, hp_search_utils, train_utils

def parse_args() -> Namespace:
    parser = ArgumentParser(description='')
    parser.add_argument("--config", type=str, required=True, help='Path to training config file')
    parser.add_argument("--hparam_config", type=str, required=True, help='Path to hyperparameter config file')
    parser.add_argument("--model", type=str, required=True, help='Model to use for training')
    parser.add_argument("--seed", type=int, default=42, help='Seed for random number generators')
    parser.add_argument("--device", type=str, default=('cuda' if torch.cuda.is_available() else 'cpu'), help='Device to run on')
    return parser.parse_args()

def cross_validate(_config: Dict, cross_val_groups: List[str]) -> float | Tuple[float, float]:
    val_rmses = []
    if is_dual_model:
        val_edge_rmses = []
    for group_id in cross_val_groups:
        logger.log(f'Cross-validating with Group {group_id} as the test set...\n')

        train_dataset, test_dataset, val_dataset = hp_search_utils.load_datasets(group_id, _config, logger)
        model = hp_search_utils.load_model(args.model, _config, train_dataset, args.device)

        # ============ Training Phase ============
        logger.log('\nTraining model...')
        _train_config = _config['training_parameters']
        optimizer = torch.optim.Adam(model.parameters(), lr=_train_config['learning_rate'], weight_decay=_train_config['adam_weight_decay'])

        base_trainer_params = train_utils.get_trainer_config(args.model, _config)
        trainer_params = {
            'model': model,
            'dataset': train_dataset,
            'val_dataset': val_dataset,
            'optimizer': optimizer,
            'logger': None,
            'device': args.device,
            **base_trainer_params,
        }

        autoregressive_train_config = _train_config['autoregressive']
        autoregressive_enabled = autoregressive_train_config.get('enabled', False)
        trainer = trainer_factory(args.model, autoregressive_enabled, **trainer_params)

        with open(os.devnull, "w") as f, redirect_stdout(f):
            trainer.train()
        trainer.training_stats.log = logger.log  # Restore logging to console
        trainer.print_stats_summary()

        # ============ Testing Phase ============
        logger.log('\nTesting model...')

        _test_config = _config['testing_parameters']
        tester_params = {
            'model': model,
            'dataset': test_dataset,
            'rollout_start': _test_config['rollout_start'],
            'rollout_timesteps': _test_config['rollout_timesteps'],
            'include_physics_loss': False,
            'logger': None,
            'device': args.device,
        }

        if is_dual_model:
            tester = DualAutoregressiveTester(**tester_params)
        elif model.__class__.__name__ in EDGE_MODELS:
            tester = EdgeAutoregressiveTester(**tester_params)
        else:
            tester = NodeAutoregressiveTester(**tester_params)
        with open(os.devnull, "w") as f, redirect_stdout(f):
            tester.test()

        avg_rmse = tester.get_avg_node_rmse()
        if ~np.isfinite(avg_rmse):
            raise optuna.TrialPruned("NAN or Inf RMSE encountered during cross-validation.")
        val_rmses.append(avg_rmse)
        logger.log(f'Group {group_id} RMSE: {avg_rmse:.4e}')

        if is_dual_model:
            avg_edge_rmse = tester.get_avg_edge_rmse()
            if ~np.isfinite(avg_edge_rmse):
                raise optuna.TrialPruned("NAN or Inf Edge RMSE encountered during cross-validation.")
            val_edge_rmses.append(avg_edge_rmse)
            logger.log(f'Group {group_id} Edge RMSE: {avg_edge_rmse:.4e}')

    val_rmses = np.array(val_rmses)
    avg_val_rmse = val_rmses.mean()
    logger.log(f'\nAverage RMSE across all events: {avg_val_rmse:.4e}')
    if not is_dual_model:
        return avg_val_rmse

    val_edge_rmses = np.array(val_edge_rmses)
    avg_val_edge_rmse = val_edge_rmses.mean()
    logger.log(f'Average Edge RMSE across all events: {avg_val_edge_rmse:.4e}')
    return avg_val_rmse, avg_val_edge_rmse

def create_objective(cross_val_groups: List[str]):
    def objective(trial: optuna.Trial) -> float:
        hyperparamters = hparam_config['hyperparameters']
        updated_config = hp_search_utils.suggest_hyperparamters(trial, hyperparamters, config, logger)
        return cross_validate(updated_config, cross_val_groups)
    return objective

def plot_hyperparameter_search_results(study: optuna.Study):
    output_dir = hparam_config['output_dir']
    if output_dir is None:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    node_plot_params = {
        'study': study,
        'target_name': 'Node RMSE',
        'target': lambda t: t.values[0] if is_dual_model else None,
    }

    fig = plot_optimization_history(**node_plot_params)
    fig.write_html(os.path.join(output_dir, f'{study.study_name}_optimization_history.html'))

    fig = plot_slice(**node_plot_params)
    fig.write_html(os.path.join(output_dir, f'{study.study_name}_slice_plot.html'))

    if is_dual_model:
        fig = plot_optimization_history(study, target=lambda t: t.values[1], target_name='Edge RMSE')
        fig.write_html(os.path.join(output_dir, f'{study.study_name}_edge_optimization_history.html'))

        fig = plot_slice(study, target=lambda t: t.values[1], target_name='Edge RMSE')
        fig.write_html(os.path.join(output_dir, f'{study.study_name}_edge_slice_plot.html'))

        fig = plot_pareto_front(study=study, target_names=['Node RMSE', 'Edge RMSE'])
        fig.write_html(os.path.join(output_dir, f'{study.study_name}_pareto_front.html'))

def main():
    try:
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            logger.log(f'Setting random seed to {args.seed}')

        current_device = torch.cuda.get_device_name(args.device) if args.device != 'cpu' else 'CPU'
        logger.log(f'Using device: {current_device}')

        # Begin hyperparameter search
        dataset_parameters = config['dataset_parameters']
        root_dir = dataset_parameters['root_dir']
        dataset_summary_file = dataset_parameters['training']['dataset_summary_file']
        early_stopping_patience = train_config['early_stopping_patience']
        percent_validation = train_config['val_split_percent'] if early_stopping_patience is not None else None
        if early_stopping_patience is not None:
            assert percent_validation is not None, 'Validation split percentage must be specified if early stopping is used.'
        num_folds = hparam_config['num_folds']
        logger.log(f'Creating {num_folds}-fold cross-validation dataset files from {dataset_summary_file}...')
        cross_val_groups, temp_dir_paths = hp_search_utils.create_cross_val_dataset_files(root_dir,
                                                                                          dataset_summary_file,
                                                                                          num_folds,
                                                                                          percent_validation)

        study_name = f'{"_".join(hparam_config['hyperparameters'].keys())}'
        study_kwargs = {'study_name': study_name }
        if is_dual_model:
            study_kwargs['directions'] = ['minimize', 'minimize']
        else:
            study_kwargs['direction'] = 'minimize'
        num_trials = hparam_config['num_trials']
        study = optuna.create_study(**study_kwargs)
        logger.log(f'Using sampler: {study.sampler.__class__.__name__ if study.sampler else None}')
        logger.log(f'Using pruner: {study.pruner.__class__.__name__ if study.pruner else None}')

        objective = create_objective(cross_val_groups)
        logger.log(f'Running hyperparameter search for {num_trials} trials...')
        study.optimize(objective, n_trials=num_trials)

        if is_dual_model:
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
    except Exception:
        logger.log(f'Unexpected error:\n{traceback.format_exc()}')
    finally:
        if 'temp_dir_paths' in locals():
            # Clean up temporary directories
            file_utils.delete_temp_dirs(temp_dir_paths)

if __name__ == '__main__':
    args = parse_args()
    config = file_utils.read_yaml_file(args.config)
    hparam_config = file_utils.read_yaml_file(args.hparam_config)
    config['model_parameters'] = {args.model: config['model_parameters'][args.model]}

    # Initialize logger
    train_config = config['training_parameters']
    log_path = train_config['log_path']
    logger = Logger(log_path=log_path)

    logger.log('================================================')
    logger.log(f'Running Hyperparameter Search with {args.model} model')
    logger.log(f'Configuration: {pformat(config)}')
    logger.log(f'Hyperparameter Search Configuration: {pformat(hparam_config)}')

    is_dual_model = 'NodeEdgeGNN' in args.model

    main()

    logger.log('================================================')
