import numpy as np

from datetime import datetime
from utils.file_utils import read_hdf_file_as_numpy

def get_event_timesteps(filepath: str) -> np.ndarray:
    property_path = 'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.Time Date Stamp'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)

    def format(x: np.bytes_) -> datetime:
        TIMESTAMP_FORMAT = '%d%b%Y %H:%M:%S'
        time_str = x.decode('UTF-8')
        time_stamp = datetime.strptime(time_str, TIMESTAMP_FORMAT)
        return time_stamp

    vec_format = np.vectorize(format)
    time_series = vec_format(data)
    return time_series

def get_cell_area(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Geometry.2D Flow Areas.{perimeter_name}.Cells Surface Area'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_min_cell_elevation(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Geometry.2D Flow Areas.{perimeter_name}.Cells Minimum Elevation'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_roughness(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f"Geometry.2D Flow Areas.{perimeter_name}.Cells Center Manning's n"
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_cumulative_rainfall(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    try:
        property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.2D Flow Areas.{perimeter_name}.Cell Cumulative Precipitation Depth'
        data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
        data = data.astype(dtype)
        area = get_cell_area(filepath, perimeter_name, dtype=dtype)
        data = (data / 1000) * area  # Convert mm to m³
    except KeyError:
        # Rainfall data not available in the file
        water_level = get_water_level(filepath, perimeter_name)
        data = np.zeros_like(water_level, dtype=dtype)

    return data

def get_water_level(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.2D Flow Areas.{perimeter_name}.Water Surface'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_water_volume(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.2D Flow Areas.{perimeter_name}.Cell Volume'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_edge_direction_x(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Geometry.2D Flow Areas.{perimeter_name}.Faces NormalUnitVector and Length'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    data = data.astype(dtype)

    # Get x direction property
    data = np.squeeze(np.take(data, indices=[0], axis=1))

    return data

def get_edge_direction_y(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Geometry.2D Flow Areas.{perimeter_name}.Faces NormalUnitVector and Length'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    data = data.astype(dtype)

    # Get y direction property
    data = np.squeeze(np.take(data, indices=[1], axis=1))

    return data

def get_face_length(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Geometry.2D Flow Areas.{perimeter_name}.Faces NormalUnitVector and Length'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    data = data.astype(dtype)

    # Get face length property
    data = np.squeeze(np.take(data, indices=[2], axis=1))

    return data

def get_velocity(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.2D Flow Areas.{perimeter_name}.Face Velocity'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_face_flow(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.2D Flow Areas.{perimeter_name}.Face Flow'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_inflow(filepath: str, perimeter_name: str = 'Perimeter 1', dtype: np.dtype = np.float32) -> np.ndarray:
    property_path = f'Results.Unsteady.Output.Output Blocks.Base Output.Unsteady Time Series.2D Flow Areas.{perimeter_name}.Boundary Conditions.UP_BC - Flow per Face'
    data = read_hdf_file_as_numpy(filepath=filepath, property_path=property_path)
    return data.astype(dtype)

def get_wl_vol_interp_points_for_cell(cell_idx: int, filepath: str, perimeter_name: str = 'Perimeter 1') -> np.ndarray:
    info_property_path = f'Geometry.2D Flow Areas.{perimeter_name}.Cells Volume Elevation Info'
    info_data = read_hdf_file_as_numpy(filepath=filepath, property_path=info_property_path)

    values_property_path = f'Geometry.2D Flow Areas.{perimeter_name}.Cells Volume Elevation Values'
    values_data = read_hdf_file_as_numpy(filepath=filepath, property_path=values_property_path)

    assert cell_idx < info_data.shape[0], "Cell is not found in the data."
    start_idx, count = info_data[cell_idx]
    assert count > 1, "Cell has no elevation-volume points."
    link_data = values_data[start_idx:start_idx+count]
    water_level = link_data[:, 0]
    volume = link_data[:, 1]
    return water_level, volume
