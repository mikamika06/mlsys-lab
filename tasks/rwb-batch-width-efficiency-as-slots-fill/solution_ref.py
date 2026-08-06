import numpy as np


def batch_width_utilization(occupancy: np.ndarray, N: int) -> dict:
  """Per-step and mean batch-width utilization from an occupancy trace."""
  per_step_list = []
  total = 0.0
  for val in occupancy:
    item = float(val) / N
    per_step_list.append(item)
    total += item
  per_step = np.array(per_step_list, dtype=np.float64)
  mean = total / len(occupancy)
  return {"per_step": per_step, "mean": mean}
