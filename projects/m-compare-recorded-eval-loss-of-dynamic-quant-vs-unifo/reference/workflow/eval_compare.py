def compare_eval_loss(dyn_logs, uni_logs):
  d_map = {item["step"]: item["eval_loss"] for item in dyn_logs}
  u_map = {item["step"]: item["eval_loss"] for item in uni_logs}
  common_steps = sorted(set(d_map.keys()) & set(u_map.keys()))
  if not common_steps:
    return 0.0
  rel_diffs = [abs(d_map[s] - u_map[s]) / u_map[s] for s in common_steps]
  return sum(rel_diffs) / len(rel_diffs)
