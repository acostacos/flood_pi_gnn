
from data import AutoregressiveFloodDataset
from torch.optim.lr_scheduler import StepLR
from typing import Optional

from .base_trainer import BaseTrainer

class BaseAutoregressiveTrainer(BaseTrainer):
    def __init__(self,
                 init_num_timesteps: int = 1,
                 total_num_timesteps: int = 1,
                 timestep_increment: int = 1,
                 learning_rate_decay: float = 0.7,
                 max_curriculum_epochs: Optional[int] = None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.init_num_timesteps = init_num_timesteps
        self.total_num_timesteps = total_num_timesteps
        self.timestep_increment = timestep_increment
        self.max_curriculum_epochs = max_curriculum_epochs
        self.lr_scheduler = StepLR(self.optimizer, step_size=1, gamma=learning_rate_decay)

        assert isinstance(self.dataloader.dataset, AutoregressiveFloodDataset), 'dataset (for training) must be an instance of AutoregressiveFloodEventDataset.'
        assert hasattr(self, 'early_stopping'), 'Early stopping must be enabled for autoregressive training.'
        assert self.init_num_timesteps <= self.total_num_timesteps, 'Initial number of timesteps must be less than or equal to total number of timesteps.'
        if self.init_num_timesteps > 1 and self.num_epochs_dyn_loss > 0:
            self.training_stats.log('WARNING: not starting with a timestep of 1 for autoregressive training, while adjusting loss scaling ratios dynamically is enabled. This may lead to unexpected behavior.')
