import torch
import numpy as np

from constants import TEST_LOCAL_MASS_LOSS_NODES
from torch_geometric.loader import DataLoader
from utils.validation_stats import ValidationStats
from utils import physics_utils

from .base_tester import BaseTester

class DualRegressionTester(BaseTester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test(self):
        for event_idx, run_id in enumerate(self.dataset.hec_ras_run_ids):
            self.log(f'Validating on run {event_idx + 1}/{len(self.dataset.hec_ras_run_ids)} with Run ID {run_id}')

            validation_stats = ValidationStats(logger=self.logger,
                                                normalizer=self.dataset.normalizer,
                                                is_normalized=self.dataset.is_normalized,
                                                delta_t=self.dataset.timestep_interval)
            self.run_test_for_event(event_idx, validation_stats)
            validation_stats.print_stats_summary()
            self.events_validation_stats.append(validation_stats)

        self.log(f'Average Node RMSE across events: {self.get_avg_node_rmse():.4e}')
        self.log(f'Average Edge RMSE across events: {self.get_avg_edge_rmse():.4e}')
        if self.include_physics_loss:
            self.log(f'Average Global Mass Conservation Loss across events: {self.get_avg_global_mass_loss():.4e}')
            self.log(f'Average Local Mass Conservation Loss across events: {self.get_avg_local_mass_loss():.4e}')

    def run_test_for_event(self, event_idx: int, validation_stats: ValidationStats):
        validation_stats.start_validate()
        self.model.eval()
        with torch.no_grad():
            event_start_idx = self.dataset.event_start_idx[event_idx] + self.rollout_start
            event_end_idx = self.dataset.event_start_idx[event_idx + 1] if event_idx + 1 < len(self.dataset.event_start_idx) else self.dataset.total_rollout_timesteps
            if self.rollout_timesteps is not None:
                event_end_idx = event_start_idx + self.rollout_timesteps
                dataset_event_length = self.dataset.event_start_idx[event_idx + 1] if event_idx + 1 < len(self.dataset.event_start_idx) else self.dataset.total_rollout_timesteps
                assert event_end_idx <= dataset_event_length, \
                    f'Rollout length {event_end_idx} exceeds event length {dataset_event_length} for event {self.dataset.hec_ras_run_ids[event_idx]}.'
            event_dataset = self.dataset[event_start_idx:event_end_idx]
            dataloader = DataLoader(event_dataset, batch_size=1, shuffle=False) # Enforce batch size = 1 for autoregressive testing

            for graph in dataloader:
                graph = graph.to(self.device)

                x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
                pred_diff, edge_pred_diff = self.model(x, edge_index, edge_attr)

                # Override boundary conditions in predictions
                pred_diff[self.boundary_nodes_mask] = graph.y[self.boundary_nodes_mask]
                edge_pred_diff[self.boundary_edges_mask] = graph.y_edge[self.boundary_edges_mask]

                pred = x[:, [self.end_node_target_idx-1]] + pred_diff
                edge_pred = edge_attr[:, [self.end_edge_target_idx-1]] + edge_pred_diff

                if self.include_physics_loss:
                    # Requires normalized prediction for physics-informed loss
                    prev_node_pred, prev_edge_pred = physics_utils.get_physics_info_node_edge(x, edge_attr, self.dataset.previous_timesteps, graph)
                    validation_stats.update_physics_informed_stats_for_timestep(pred, prev_node_pred, prev_edge_pred, graph, TEST_LOCAL_MASS_LOSS_NODES)

                label = x[:, [self.end_node_target_idx-1]] + graph.y
                if self.dataset.is_normalized:
                    pred = self.dataset.normalizer.denormalize(self.dataset.NODE_TARGET_FEATURE, pred)
                    label = self.dataset.normalizer.denormalize(self.dataset.NODE_TARGET_FEATURE, label)

                # Ensure water volume is non-negative
                pred = torch.clip(pred, min=0)
                label = torch.clip(label, min=0)

                # Filter boundary conditions for metric computation
                pred = pred[self.non_boundary_nodes_mask]
                label = label[self.non_boundary_nodes_mask]

                validation_stats.update_stats_for_timestep(pred.cpu(),
                                                           label.cpu(),
                                                           water_threshold=self.threshold_per_cell,
                                                           timestamp=graph.timestep if hasattr(graph, 'timestep') else None)

                label_edge = edge_attr[:, [self.end_edge_target_idx-1]] + graph.y_edge
                if self.dataset.is_normalized:
                    edge_pred = self.dataset.normalizer.denormalize(self.dataset.EDGE_TARGET_FEATURE, edge_pred)
                    label_edge = self.dataset.normalizer.denormalize(self.dataset.EDGE_TARGET_FEATURE, label_edge)

                validation_stats.update_edge_stats_for_timestep(edge_pred.cpu(), label_edge.cpu())


        validation_stats.end_validate()

    def get_avg_edge_rmse(self) -> float:
        edge_rmses = [stat.get_avg_edge_rmse() for stat in self.events_validation_stats]
        return np.mean(edge_rmses) if edge_rmses else 0.0
