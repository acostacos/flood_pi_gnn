import torch
import numpy as np

from data.hecras_data_retrieval import get_wl_vol_interp_points_for_cell, get_cell_area
from torch import Tensor
from torch.nn.functional import mse_loss, l1_loss

def RMSE(pred: Tensor, target: Tensor) -> Tensor:
    return torch.sqrt(mse_loss(pred, target))

def MAE(pred: Tensor, target: Tensor) -> Tensor:
    return l1_loss(pred, target)

def NSE(pred: Tensor, target: Tensor) -> Tensor:
    '''Nash Sutcliffe Efficiency'''
    model_sse = torch.sum((target - pred)**2)
    mean_model_sse = torch.sum((target - target.mean())**2)
    return 1 - (model_sse / mean_model_sse)

def CSI(binary_pred: Tensor, binary_target: Tensor):
    TP = (binary_pred & binary_target).sum() #true positive
    # TN = (~binary_pred & ~binary_target).sum() #true negative
    FP = (binary_pred & ~binary_target).sum() #false positive
    FN = (~binary_pred & binary_target).sum() #false negative

    return TP / (TP + FN + FP)

def interpolate_wl_from_vol(water_volume: np.ndarray, hec_ras_file_path: str, num_nodes: int = None):
    if num_nodes is None:
        num_nodes = water_volume.shape[1]

    interp_values_cache = {}
    area = get_cell_area(hec_ras_file_path)

    num_timesteps = water_volume.shape[0]
    water_level = np.zeros_like(water_volume)
    for t in range(num_timesteps):
        for cell_idx in range(num_nodes):
            if cell_idx not in interp_values_cache:
                interp_values_cache[cell_idx] = get_wl_vol_interp_points_for_cell(cell_idx, hec_ras_file_path)
            water_level_interp, volume_interp = interp_values_cache[cell_idx]

            max_vol_interp = volume_interp.max()
            curr_vol = water_volume[t, cell_idx]
            if curr_vol <= max_vol_interp:
                # Interpolation within the range, assume water_level_interp and volume_interp are sorted in ascending order
                interpolated_wl = np.interp(curr_vol, volume_interp, water_level_interp)
            else:
                # Extrapolation beyond the maximum known elevation using linear approximation
                max_wl = water_level_interp[-1]
                delta_vol = curr_vol - max_vol_interp
                interpolated_wl = max_wl + (delta_vol / area[cell_idx])
            water_level[t, cell_idx] = interpolated_wl

        if t % 100 == 0:
            print(f'Completed interpolation for timestep {t}/{num_timesteps}')

    return water_level
