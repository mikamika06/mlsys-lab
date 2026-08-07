import ref
import numpy as np

def check(workdir):
    from routing.metrics import compute_tail_latency, get_per_replica_counts
    out = {"tail_lat_rel_err": 1.0}
    num_replicas = 4
    rng = np.random.RandomState(123)
    reqs = list(np.cumsum(rng.exponential(scale=0.5, size=200)))
    serv = list(rng.exponential(scale=2.0, size=200) + 0.1)

    ref_p99 = ref.compute_tail_latency(num_replicas, reqs, serv, strategy="power_of_two", percentile=99, seed=42)
    ref_counts = ref.get_per_replica_counts(num_replicas, reqs, serv, strategy="power_of_two", seed=42)

    try:
        got_p99 = compute_tail_latency(num_replicas, reqs, serv, strategy="power_of_two", percentile=99, seed=42)
        got_counts = get_per_replica_counts(num_replicas, reqs, serv, strategy="power_of_two", seed=42)
    except Exception as e:
        out["_note"] = f"metrics raised error: {e}"
        return out

    rel_err = abs(got_p99 - ref_p99) / abs(ref_p99) if ref_p99 != 0 else (0.0 if got_p99 == 0 else 1.0)
    counts_match = (ref_counts == got_counts)
    if counts_match and rel_err <= 0.05:
        out["tail_lat_rel_err"] = float(rel_err)
    else:
        out["_note"] = f"rel_err {rel_err}, counts_match {counts_match}"
    return out
