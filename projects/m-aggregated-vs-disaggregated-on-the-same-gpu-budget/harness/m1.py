import ref


def check(workdir):
    out = {"simulation_matched": 0.0}
    try:
        from disagg.simulator import simulate_aggregated, simulate_disaggregated
    except ImportError as e:
        out["_note"] = f"Import error: {e}"
        return out

    ref_agg, ref_disagg, _ = ref.get_reference_results()
    reqs = ref.generate_workload(42, 50)

    got_agg = simulate_aggregated(reqs, num_gpus=8, prefill_rate=5000.0, decode_rate=200.0)
    got_disagg = simulate_disaggregated(
        reqs, num_prefill_gpus=2, num_decode_gpus=6, prefill_rate=5000.0, decode_rate=200.0, kv_transfer_rate=10e7
    )

    def matches(a_list, b_list):
        if len(a_list) != len(b_list):
            return False
        for a, b in zip(a_list, b_list):
            if a["req_id"] != b["req_id"]:
                return False
            if abs(a["ttft"] - b["ttft"]) > 1e-4 or abs(a["end_time"] - b["end_time"]) > 1e-4:
                return False
        return True

    if matches(got_agg, ref_agg) and matches(got_disagg, ref_disagg):
        out["simulation_matched"] = 1.0
    else:
        out["_note"] = "Simulation results do not match reference implementation"

    return out
