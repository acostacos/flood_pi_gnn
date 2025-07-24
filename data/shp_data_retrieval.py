import numpy as np
from itertools import permutations

from utils.file_utils import read_shp_file, read_shp_file_as_numpy

def get_edge_index(filepath: str) -> np.ndarray:
    columns = ['from_node', 'to_node']
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    # Convert to edge index format
    return data.astype(np.int64).transpose()

def get_cell_elevation(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    columns = 'Elevation1'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_node_edge_index(filepath: str) -> np.ndarray:
    def str_to_list(s: str) -> list:
        clean = s.replace('[', '').replace(']', '').replace(' ', '')
        return [int(i) for i in clean.split(',')] if clean else []

    columns = ['CC_index', 'from_links', 'to_links']
    data = read_shp_file(filepath=filepath, columns=columns)
    node_edge_index = []
    node_edge_attr = []
    for _, row in data.iterrows():
        from_links = str_to_list(row['from_links'])
        to_links = str_to_list(row['to_links'])
        connected_edges = [*from_links, *to_links]
        perm = list(permutations(connected_edges, 2))
        node_edge_index.extend(perm)
        node_edge_attr.extend([row['CC_index']] * len(perm))

    node_edge_index = np.array(node_edge_index, dtype=np.int64).transpose()
    node_edge_attr = np.array(node_edge_attr, dtype=np.int64)

    return node_edge_index, node_edge_attr

def get_edge_length(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    columns = 'length'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)

def get_edge_slope(filepath: str, dtype: np.dtype = np.float32) -> np.ndarray:
    columns = 'slope'
    data = read_shp_file_as_numpy(filepath=filepath, columns=columns)
    return data.astype(dtype)
