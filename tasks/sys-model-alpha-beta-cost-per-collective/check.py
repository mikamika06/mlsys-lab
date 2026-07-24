import math
import numpy as np


def _ref_cost(collective, num_ranks, message_bytes, alpha, beta):
    """Real oracle: the standard ring-algorithm (for allgather / reduce_scatter
    / alltoall / allreduce == reduce_scatter + allgather) and binomial-tree
    (for broadcast) alpha-beta cost model.

    alpha = per-message latency (seconds), beta = per-byte transfer time
    (seconds/byte), message_bytes = total logical payload size for the
    collective (e.g. the fully-gathered size for allgather, the
    pre-reduction size for reduce_scatter).
    """
    P = num_ranks
    M = message_bytes
    frac = (P - 1) / P if P > 0 else 0.0

    if collective == "reduce_scatter" or collective == "allgather" or collective == "alltoall":
        steps = max(P - 1, 0)
        latency_term = steps * alpha
        bandwidth_term = frac * M * beta
    elif collective == "allreduce":
        steps = 2 * max(P - 1, 0)
        latency_term = steps * alpha
        bandwidth_term = 2.0 * frac * M * beta
    elif collective == "broadcast":
        steps = math.ceil(math.log2(P)) if P > 1 else 0
        latency_term = steps * alpha
        bandwidth_term = steps * M * beta
    else:
        raise ValueError(f"unknown collective {collective!r}")

    return {
        "latency_term": float(latency_term),
        "bandwidth_term": float(bandwidth_term),
        "total": float(latency_term + bandwidth_term),
    }


def _cases():
    cases = []
    collectives = ["allreduce", "allgather", "reduce_scatter", "broadcast", "alltoall"]
    configs = [
        (2, 1024.0, 1e-6, 1e-9),
        (4, 1_000_000.0, 5e-6, 4e-10),
        (8, 65536.0, 2e-6, 1e-9),
        (16, 12_345_678.0, 1e-5, 2.5e-10),
        (1, 4096.0, 1e-6, 1e-9),
        (3, 999.0, 3e-6, 8e-10),
        (32, 2_048_000.0, 8e-6, 1.2e-9),
    ]
    for c in collectives:
        for P, M, alpha, beta in configs:
            cases.append((c, P, M, alpha, beta))
    return cases


def grade(sol, fx) -> dict:
    worst_rel = 0.0
    for collective, P, M, alpha, beta in _cases():
        ref = _ref_cost(collective, P, M, alpha, beta)
        try:
            got = sol.collective_cost(collective, P, M, alpha, beta)
            lat = float(got["latency_term"])
            bw = float(got["bandwidth_term"])
            tot = float(got["total"])
        except Exception:
            return {"modeled_mem_access": float("inf")}

        if not (np.isfinite(lat) and np.isfinite(bw) and np.isfinite(tot)):
            return {"modeled_mem_access": float("inf")}

        for got_v, ref_v in ((lat, ref["latency_term"]), (bw, ref["bandwidth_term"]), (tot, ref["total"])):
            denom = abs(ref_v) + 1e-12
            rel = abs(got_v - ref_v) / denom
            worst_rel = max(worst_rel, rel)

    return {"modeled_mem_access": worst_rel}
