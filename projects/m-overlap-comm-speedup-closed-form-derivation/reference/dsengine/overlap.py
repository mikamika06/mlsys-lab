def compute_step_time(t_comp: float, t_comm: float, overlap_factor: float) -> float:
    if t_comp < 0 or t_comm < 0:
        raise ValueError("Time components must be non-negative")
    overlap_factor = max(0.0, min(1.0, float(overlap_factor)))
    exposed_comm = max(0.0, t_comm - overlap_factor * t_comp)
    return float(t_comp + exposed_comm)


def compute_speedup(t_comp: float, t_comm: float, overlap_factor: float) -> float:
    unoverlapped_time = t_comp + t_comm
    if unoverlapped_time == 0:
        return 1.0
    overlapped_time = compute_step_time(t_comp, t_comm, overlap_factor)
    return float(unoverlapped_time / overlapped_time)


def min_overlap_for_speedup(t_comp: float, t_comm: float, target_speedup: float) -> float:
    if target_speedup < 1.0:
        return 0.0
    unoverlapped = t_comp + t_comm
    if unoverlapped == 0:
        return 0.0
    max_possible_speedup = unoverlapped / max(t_comp, t_comm) if max(t_comp, t_comm) > 0 else 1.0
    if target_speedup > max_possible_speedup + 1e-9:
        return -1.0
    target_step_time = unoverlapped / target_speedup
    req_exposed = target_step_time - t_comp
    if req_exposed >= t_comm:
        return 0.0
    if t_comp == 0:
        return 1.0 if req_exposed < t_comm else 0.0
    required_factor = (t_comm - req_exposed) / t_comp
    return float(max(0.0, min(1.0, required_factor)))
