import numpy as np


def preprocess_app_side(raw_frames: np.ndarray, mean: list, std: list) -> np.ndarray:
    raise NotImplementedError


def preprocess_in_graph_node(raw_frames: np.ndarray, mean: list, std: list) -> np.ndarray:
    raise NotImplementedError


def compare_pipeline_memory(batch_size: int, height: int, width: int, channels: int) -> dict:
    raise NotImplementedError
