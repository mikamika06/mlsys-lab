import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from fsdp_analyzer.communication import compute_per_step_communication_bytes
    from fsdp_analyzer.memory import compute_layer_transient_peak_memory_bytes

    out = {"comm_matched": 0.0, "peak_memory_matched": 0.0}

    comm_ok = True
    strategies = ["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"]
    for cfg in ref.COMM_CONFIGS:
        for strat in strategies:
            want = ref.ref_compute_per_step_communication_bytes(
                cfg["num_params"],
                cfg["world_size"],
                strat,
                cfg["bytes_per_param"],
                cfg["bytes_per_grad"],
            )
            got = compute_per_step_communication_bytes(
                cfg["num_params"],
                cfg["world_size"],
                strat,
                cfg["bytes_per_param"],
                cfg["bytes_per_grad"],
            )
            if want != got:
                comm_ok = False
                out["_note"] = f"comm mismatch strategy {strat}: got {got}, want {want}"
                break
        if not comm_ok:
            break

    if comm_ok:
        out["comm_matched"] = 1.0

    mem_ok = True
    for cfg in ref.MEMORY_CONFIGS:
        want = ref.ref_compute_layer_transient_peak_memory_bytes(**cfg)
        got = compute_layer_transient_peak_memory_bytes(**cfg)
        if want != got:
            mem_ok = False
            out["_note"] = f"peak memory mismatch: got {got}, want {want}"
            break

    if mem_ok:
        out["peak_memory_matched"] = 1.0

    return out
