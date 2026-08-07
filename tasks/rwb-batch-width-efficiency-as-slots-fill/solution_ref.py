def batch_width_utilization(occupancy: list[int], N: int) -> dict:
  """Per-step and mean batch-width utilization from an occupancy trace."""
  per_step = []
  total = 0.0
  for val in occupancy:
    item = float(val) / N
    per_step.append(item)
    total += item
  mean = total / len(occupancy)
  return {"per_step": per_step, "mean": mean}
