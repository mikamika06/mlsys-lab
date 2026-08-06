import numpy as np
from fused_grad.backward import _fused_op_grad


def classify_atomic_requirement(index_map: np.ndarray) -> bool:
    """Classify whether atomic operations are required based on index overlap."""
    return len(index_map) != len(np.unique(index_map))


def simulate_parallel_backward(
    grad_output: np.ndarray,
    x: np.ndarray,
    index_map: np.ndarray,
    atomic: bool,
    num_threads: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Simulate parallel backward execution with or without atomic updates."""
    m = len(index_map)
    n = len(x)
    grad_x = np.zeros(n, dtype=np.float64)
    ds = _fused_op_grad(x)

    if atomic:
        for i in range(m):
            j = index_map[i]
            grad_x[j] += grad_output[i] * ds[j]
        return grad_x

    rng = np.random.default_rng(seed)
    read_times = rng.random(m)
    write_times = read_times + rng.random(m) + 0.1

    events = []
    for i in range(m):
        events.append((read_times[i], "read", i))
        events.append((write_times[i], "write", i))
    events.sort(key=lambda item: item[0])

    thread_regs = np.zeros(m, dtype=np.float64)
    for _, event_type, i in events:
        j = index_map[i]
        delta = grad_output[i] * ds[j]
        if event_type == "read":
            thread_regs[i] = grad_x[j]
        else:
            grad_x[j] = thread_regs[i] + delta

    return grad_x


def analyze_determinism(
    x: np.ndarray, index_map: np.ndarray, grad_output: np.ndarray, num_runs: int = 10
) -> dict:
    """Analyze determinism and maximum error across multiple simulated runs."""
    req_atomic = classify_atomic_requirement(index_map)

    atomic_runs = [
        simulate_parallel_backward(grad_output, x, index_map, atomic=True, seed=r)
        for r in range(num_runs)
    ]
    non_atomic_runs = [
        simulate_parallel_backward(grad_output, x, index_map, atomic=False, seed=r)
        for r in range(num_runs)
    ]

    atomic_is_det = all(np.allclose(atomic_runs[0], r) for r in atomic_runs)
    non_atomic_is_det = all(np.allclose(non_atomic_runs[0], r) for r in non_atomic_runs)

    ref_bwd = atomic_runs[0]
    max_err = max(float(np.max(np.abs(r - ref_bwd))) for r in non_atomic_runs)

    return {
        "requires_atomic": req_atomic,
        "atomic_is_deterministic": atomic_is_det,
        "non_atomic_is_deterministic": non_atomic_is_det,
        "non_atomic_max_error": max_err,
    }
