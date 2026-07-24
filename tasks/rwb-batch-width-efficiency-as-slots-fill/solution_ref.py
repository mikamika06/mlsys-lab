import numpy as np


def batch_width_utilization(occupancy: np.ndarray, N: int) -> dict:
    """Per-step and mean batch-width utilization from an occupancy trace."""
    per_step = np.asarray(occupancy, dtype=np.float64) / N
    mean = float(np.mean(per_step))
    return {"per_step": per_step, "mean": mean}
