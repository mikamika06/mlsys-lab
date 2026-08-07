import numpy as np


def sample_frames(batch_size=4, h=32, w=32, c=3, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=(batch_size, h, w, c), dtype=np.uint8)


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def nhwc_to_nchw(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(arr, (0, 3, 1, 2)))


def nchw_to_nhwc(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(arr, (0, 2, 3, 1)))


def preprocess_app_side(raw_frames: np.ndarray, mean: list, std: list) -> np.ndarray:
    nchw = nhwc_to_nchw(raw_frames)
    m = np.array(mean, dtype=np.float32).reshape(1, -1, 1, 1)
    s = np.array(std, dtype=np.float32).reshape(1, -1, 1, 1)
    return (nchw.astype(np.float32) / 255.0 - m) / s


def preprocess_in_graph_node(raw_frames: np.ndarray, mean: list, std: list) -> np.ndarray:
    m = np.array(mean, dtype=np.float32).reshape(1, 1, 1, -1)
    s = np.array(std, dtype=np.float32).reshape(1, 1, 1, -1)
    norm_nhwc = (raw_frames.astype(np.float32) / 255.0 - m) / s
    return np.ascontiguousarray(np.transpose(norm_nhwc, (0, 3, 1, 2)))
