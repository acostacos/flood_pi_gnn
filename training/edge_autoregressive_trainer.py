import os
import time
import numpy as np
import torch

from contextlib import redirect_stdout
from torch import Tensor
from data import AutoregressiveFloodDataset
from testing import EdgeAutoregressiveTester
from utils import EarlyStopping

from .base_autoregressive_trainer import BaseAutoregressiveTrainer

class EdgeAutoregressiveTrainer(BaseAutoregressiveTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ds: AutoregressiveFloodDataset = self.dataloader.dataset
        # Get non-boundary nodes/edges and threshold for metric computation
        self.boundary_edges_mask = ds.boundary_condition.boundary_edges_mask

        # Get sliding window indices
        sliding_window_length = ds.previous_timesteps + 1
        target_edges_idx = ds.DYNAMIC_EDGE_FEATURES.index(ds.EDGE_TARGET_FEATURE)
        self.start_edge_target_idx = ds.num_static_edge_features + (target_edges_idx * sliding_window_length)
        self.end_edge_target_idx = self.start_edge_target_idx + sliding_window_length

    def train(self):
        '''Multi-step-ahead loss with curriculum learning.'''
        self.training_stats.start_train()
        current_num_timesteps = self.init_num_timesteps
        current_timestep_epochs = 0

        for epoch in range(self.num_epochs):
            train_start_time = time.time()

            edge_pred_epoch_loss = self._train_model(current_num_timesteps)

            logging_str = f'Epoch [{epoch + 1}/{self.num_epochs}]\n'
            logging_str += f'\tEdge Prediction Loss: {edge_pred_epoch_loss:.4e}'
            self.training_stats.log(logging_str)

            self.training_stats.add_loss(edge_pred_epoch_loss)
            self.training_stats.add_loss_component('edge_prediction_loss', edge_pred_epoch_loss)

            train_end_time = time.time()
            train_duration = train_end_time - train_start_time
            self.training_stats.log(f'\tEpoch Train Duration: {train_duration:.2f} seconds')

            val_edge_rmse = self.validate()
            self.training_stats.log(f'\n\tValidation Edge RMSE: {val_edge_rmse:.4e}')
            self.training_stats.add_val_loss_component('val_edge_rmse', val_edge_rmse)

            current_timestep_epochs += 1

            is_early_stopped = self.early_stopping(val_edge_rmse, self.model)
            is_max_exceeded = self.max_curriculum_epochs is not None and current_timestep_epochs >= self.max_curriculum_epochs
            if is_early_stopped or is_max_exceeded:
                if current_num_timesteps < self.total_num_timesteps:
                    self.training_stats.log(f'\tCurriculum learning for {current_num_timesteps} steps ended after {current_timestep_epochs} epochs.')
                    current_num_timesteps += self.timestep_increment
                    current_timestep_epochs = 0
                    self.early_stopping = EarlyStopping(patience=self.early_stopping.patience)
                    self.training_stats.log(f'\tIncreased current_num_timesteps to {current_num_timesteps} timesteps.')
                    self.lr_scheduler.step()
                    self.training_stats.log(f'\tDecayed learning rate to {self.lr_scheduler.get_last_lr()[0]:.4e}.')
                    continue

                self.training_stats.log('Training completed due to early stopping.')
                break

        self.training_stats.end_train()

    def _train_model(self, current_num_timesteps: int) -> float:
        self.model.train()
        running_edge_pred_loss = 0.0

        for batch in self.dataloader:
            self.optimizer.zero_grad()

            batch = batch.to(self.device)
            edge_attr, edge_index = batch.edge_attr[:, :, 0], batch.edge_index

            total_batch_loss = 0.0
            total_batch_edge_pred_loss = 0.0
            edge_sliding_window = edge_attr[:, self.start_edge_target_idx:self.end_edge_target_idx].clone()
            for i in range(current_num_timesteps):
                x, edge_attr = batch.x[:, :, i], batch.edge_attr[:, :, i]

                # Override graph data with sliding window
                edge_attr = torch.concat([edge_attr[:, :self.start_edge_target_idx], edge_sliding_window, edge_attr[:, self.end_edge_target_idx:]], dim=1)

                edge_pred_diff = self.model(x, edge_index, edge_attr)
                edge_pred_diff = self._override_pred_bc(edge_pred_diff, batch, i)

                loss = self._compute_edge_loss(edge_pred_diff, batch, i)
                total_batch_edge_pred_loss += loss.item()

                total_batch_loss = total_batch_loss + loss

                if i < current_num_timesteps - 1:  # Don't update on last iteration
                    edge_pred = edge_sliding_window[:, [-1]] + edge_pred_diff
                    next_edge_sliding_window = torch.cat((edge_sliding_window[:, 1:], edge_pred), dim=1)

                    edge_sliding_window = next_edge_sliding_window

            avg_batch_loss = total_batch_loss / current_num_timesteps
            avg_batch_loss.backward()
            self._clip_gradients()
            self.optimizer.step()

            # Loss Updates
            running_edge_pred_loss += total_batch_edge_pred_loss / current_num_timesteps

        edge_pred_epoch_loss = running_edge_pred_loss / len(self.dataloader)

        return edge_pred_epoch_loss

    def validate(self):
        val_tester = EdgeAutoregressiveTester(
            model=self.model,
            dataset=self.val_dataset,
            include_physics_loss=False,
            device=self.device
        )
        with open(os.devnull, "w") as f, redirect_stdout(f):
            val_tester.test()

        edge_rmse = val_tester.get_avg_edge_rmse()
        return edge_rmse

    def _compute_edge_loss(self, edge_pred: Tensor, batch, timestep: int) -> Tensor:
        label = batch.y_edge[:, :, timestep]
        return self.loss_func(edge_pred, label)

    def _override_pred_bc(self, edge_pred: Tensor, batch, timestep: int) -> Tensor:
        batch_boundary_edges_mask = np.tile(self.boundary_edges_mask, batch.num_graphs)
        edge_pred[batch_boundary_edges_mask] = batch.y_edge[batch_boundary_edges_mask, :, timestep]
        return edge_pred
