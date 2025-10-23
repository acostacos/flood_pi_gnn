import os
import time
import numpy as np
import torch

from contextlib import redirect_stdout
from torch import Tensor
from data import AutoregressiveFloodDataset
from testing import NodeAutoregressiveTester
from typing import Tuple
from utils import EarlyStopping, physics_utils, train_utils

from .base_autoregressive_trainer import BaseAutoregressiveTrainer
from .physics_informed_trainer import PhysicsInformedTrainer

class NodeAutoregressiveTrainer(BaseAutoregressiveTrainer, PhysicsInformedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        ds: AutoregressiveFloodDataset = self.dataloader.dataset
        # Get non-boundary nodes/edges and threshold for metric computation
        self.boundary_nodes_mask = ds.boundary_condition.boundary_nodes_mask

        # Get sliding window indices
        sliding_window_length = ds.previous_timesteps + 1
        target_nodes_idx = ds.DYNAMIC_NODE_FEATURES.index(ds.NODE_TARGET_FEATURE)
        self.start_node_target_idx = ds.num_static_node_features + (target_nodes_idx * sliding_window_length)
        self.end_node_target_idx = self.start_node_target_idx + sliding_window_length

    def train(self):
        '''Multi-step-ahead loss with curriculum learning.'''
        self.training_stats.start_train()
        current_num_timesteps = self.init_num_timesteps
        current_timestep_epochs = 0

        for epoch in range(self.num_epochs):
            train_start_time = time.time()

            train_losses = self._train_model(epoch, current_num_timesteps)
            epoch_loss, pred_epoch_loss, global_mass_epoch_loss, local_mass_epoch_loss = train_losses

            logging_str = f'Epoch [{epoch + 1}/{self.num_epochs}]\n'
            logging_str += f'\tLoss: {epoch_loss:.4e}\n'
            logging_str += f'\tNode Prediction Loss: {pred_epoch_loss:.4e}'
            self.training_stats.log(logging_str)

            self.training_stats.add_loss(epoch_loss)
            self.training_stats.add_loss_component('prediction_loss', pred_epoch_loss)

            if self.use_physics_loss:
                self._log_epoch_physics_loss(global_mass_epoch_loss, local_mass_epoch_loss)

            self._update_loss_scaler_for_epoch(epoch)

            train_end_time = time.time()
            train_duration = train_end_time - train_start_time
            self.training_stats.log(f'\tEpoch Train Duration: {train_duration:.2f} seconds')

            val_node_rmse = self.validate()
            self.training_stats.log(f'\n\tValidation Node RMSE: {val_node_rmse:.4e}')
            self.training_stats.add_val_loss_component('val_node_rmse', val_node_rmse)

            current_timestep_epochs += 1

            is_early_stopped = self.early_stopping(val_node_rmse, self.model)
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
        self._add_scaled_physics_loss_history()

    def _train_model(self, epoch: int, current_num_timesteps: int) -> Tuple[float, float]:
        self.model.train()

        running_pred_loss = 0.0
        running_global_mass_loss = 0.0
        running_local_mass_loss = 0.0

        for batch in self.dataloader:
            self.optimizer.zero_grad()

            batch = batch.to(self.device)
            x, edge_index = batch.x[:, :, 0], batch.edge_index

            total_batch_loss = 0.0
            total_batch_pred_loss = 0.0
            total_batch_global_mass_loss = 0.0
            total_batch_local_mass_loss = 0.0

            sliding_window = x[:, self.start_node_target_idx:self.end_node_target_idx].clone()
            for i in range(current_num_timesteps):
                x, edge_attr = batch.x[:, :, i], batch.edge_attr[:, :, i]

                # Override graph data with sliding window
                x = torch.concat([x[:, :self.start_node_target_idx], sliding_window, x[:, self.end_node_target_idx:]], dim=1)

                pred_diff = self.model(x, edge_index, edge_attr)
                pred_diff = self._override_pred_bc(pred_diff, batch, i)

                pred_loss = self._compute_node_loss(pred_diff, batch, i)
                pred_loss = self._scale_node_pred_loss(epoch, pred_loss)
                total_batch_pred_loss += pred_loss.item()

                step_loss = pred_loss

                previous_timesteps = self.dataloader.dataset.previous_timesteps
                prev_node_pred = sliding_window[:, [-1]]
                pred = prev_node_pred + pred_diff

                if self.use_physics_loss:
                    curr_face_flow = physics_utils.get_curr_flow_from_edge_features(edge_attr, previous_timesteps)
                    global_loss, local_loss = self._get_physics_loss(epoch, pred, prev_node_pred,
                                                                     curr_face_flow, pred_loss, batch,
                                                                     current_timestep=i)
                    total_batch_global_mass_loss += global_loss.item()
                    total_batch_local_mass_loss += local_loss.item()
                    step_loss = step_loss + global_loss + local_loss

                total_batch_loss = total_batch_loss + step_loss

                if i < current_num_timesteps - 1:  # Don't update on last iteration
                    next_sliding_window = torch.cat((sliding_window[:, 1:], pred), dim=1)

                    sliding_window = next_sliding_window

            avg_batch_loss = total_batch_loss / current_num_timesteps
            avg_batch_loss.backward()
            self._clip_gradients()
            self.optimizer.step()
        
        # Loss Updates
        total_losses = (total_batch_pred_loss, total_batch_global_mass_loss, total_batch_local_mass_loss)
        avg_losses = train_utils.divide_losses(total_losses, current_num_timesteps)
        avg_pred_loss, avg_global_mass_loss, avg_local_mass_loss = avg_losses
        
        running_pred_loss += avg_pred_loss
        running_global_mass_loss += avg_global_mass_loss
        running_local_mass_loss += avg_local_mass_loss

        running_loss = running_pred_loss + running_global_mass_loss + running_local_mass_loss
        running_losses = (running_loss, running_pred_loss, running_global_mass_loss, running_local_mass_loss)
        epoch_loss = train_utils.divide_losses(running_losses, len(self.dataloader))
        epoch_loss, pred_epoch_loss, global_mass_epoch_loss, local_mass_epoch_loss = epoch_loss

        return epoch_loss, pred_epoch_loss, global_mass_epoch_loss, local_mass_epoch_loss

    def validate(self):
        val_tester = NodeAutoregressiveTester(
            model=self.model,
            dataset=self.val_dataset,
            include_physics_loss=False,
            device=self.device
        )
        with open(os.devnull, "w") as f, redirect_stdout(f):
            val_tester.test()

        node_rmse = val_tester.get_avg_node_rmse()
        return node_rmse

    def _compute_node_loss(self, pred: Tensor, batch, timestep: int) -> Tensor:
        label = batch.y[:, :, timestep]
        return self.loss_func(pred, label)

    def _override_pred_bc(self, pred: Tensor, batch, timestep: int) -> Tensor:
        batch_boundary_nodes_mask = np.tile(self.boundary_nodes_mask, batch.num_graphs)
        pred[batch_boundary_nodes_mask] = batch.y[batch_boundary_nodes_mask, :, timestep]
        return pred
