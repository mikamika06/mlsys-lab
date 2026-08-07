def compute_step_time(t_comp: float, t_comm: float, overlap_factor: float) -> float:
    """Compute total step execution time under communication overlap."""
    raise NotImplementedError


def compute_speedup(t_comp: float, t_comm: float, overlap_factor: float) -> float:
    """Compute speedup ratio comparing unoverlapped time to overlapped time."""
    raise NotImplementedError


def min_overlap_for_speedup(t_comp: float, t_comm: float, target_speedup: float) -> float:
    """Solve for minimum overlap factor needed to achieve target speedup."""
    raise NotImplementedError
