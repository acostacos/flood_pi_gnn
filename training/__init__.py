from constants import EDGE_MODELS
from .base_trainer import BaseTrainer
from .dual_autoregressive_trainer import DualAutoregressiveTrainer
from .dual_regression_trainer import DualRegressionTrainer
from .edge_autoregressive_trainer import EdgeAutoregressiveTrainer
from .edge_regression_trainer import EdgeRegressionTrainer
from .node_autoregressive_trainer import NodeAutoregressiveTrainer
from .node_regression_trainer import NodeRegressionTrainer

def trainer_factory(model_name: str, autoregressive: bool, *args, **kwargs) -> BaseTrainer:
    if 'NodeEdgeGNN' in model_name:
        if autoregressive:
            return DualAutoregressiveTrainer(*args, **kwargs)
        return DualRegressionTrainer(*args, **kwargs)

    if model_name in EDGE_MODELS:
        if autoregressive:
            return EdgeAutoregressiveTrainer(*args, **kwargs)
        return EdgeRegressionTrainer(*args, **kwargs)

    if autoregressive:
        return NodeAutoregressiveTrainer(*args, **kwargs)
    return NodeRegressionTrainer(*args, **kwargs)

__all__ = [
    'DualAutoregressiveTrainer',
    'DualRegressionTrainer',
    'EdgeRegressionTrainer',
    'NodeAutoregressiveTrainer',
    'NodeRegressionTrainer',
    'trainer_factory',
]
