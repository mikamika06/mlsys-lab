import numpy as np
from edgeio.layout import nhwc_to_nchw


def preprocess_app_side(raw_frames: np.ndarray, mean: list, std: list) -> np.ndarray:
    """App-side: converts layout NHWC -> NCHW first, then normalizes."""
    nchw = nhwc_to_nchw(raw_frames)
    m = np.array(mean, dtype=np.float32).reshape(1, -1, 1, 1)
    s = np.array(std, dtype=np.float32).reshape(1, -1, 1, 1)
    return (nchw.astype(np.float32) / 255.0 - m) / s


def preprocess_in_graph_node(raw_frames: np.ndarray, mean: list, std: list) -> np.ndarray:
    """In-graph: normalizes on NHWC raw data, then transposes to NCHW inside the graph."""
    m = np.array(mean, dtype=np.float32).reshape(1, 1, 1, -1)
    s = np.array(std, dtype=np.float32).reshape(1, 1, 1, -1)
    norm_nhwc = (raw_frames.astype(np.float32) / 255.0 - m) / s
    return np.ascontiguousarray(np.transpose(norm_nhwc, (0, 3, 1, 2)))


def compare_pipeline_memory(batch_size: int, height: int, width: int, channels: int) -> dict:
    """Compares memory requirements and transfer overhead between app-side and in-graph strategies."""
    u8_bytes = batch_size * height * width * channels * 1
    f32_bytes = batch_size * height * width * channels * 4

    app_side_host_transfer = f32_bytes
    in_graph_host_transfer = u8_bytes

    return {
        "app_side_host_transfer_bytes": app_side_host_transfer,
        "in_graph_host_transfer_bytes": in_graph_host_transfer,
        "bandwidth_reduction_factor": float(app_side_host_transfer / in_graph_host_transfer),
    }
