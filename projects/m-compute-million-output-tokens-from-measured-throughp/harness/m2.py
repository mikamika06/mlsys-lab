import sys
import os

def check(workdir):
    sys.path.insert(0, os.path.join(workdir, "harness"))
    sys.path.insert(0, workdir)
    import ref

    out = {"replica_matches": 0.0}
    try:
        from capacity.sizing import compute_required_gpus
        from capacity.selector import select_cheapest_config
    except Exception as e:
        out["_note"] = f"Import error: {e}"
        return out

    _, traffic_cases, candidate_matrix = ref.generate_test_cases()

    for idx, case in enumerate(traffic_cases):
        want = ref.ref_compute_required_gpus(case["curve"], case["throughput"], case["gpus"], case["headroom"])
        try:
            got = compute_required_gpus(case["curve"], case["throughput"], case["gpus"], case["headroom"])
        except Exception as e:
            out["_note"] = f"Error in compute_required_gpus case {idx}: {e}"
            return out

        if got != want:
            out["_note"] = f"Sizing mismatch on case {idx}: got {got}, want {want}"
            return out

    slo_targets = [150.0, 100.0, 65.0]
    for slo in slo_targets:
        want_sel = ref.ref_select_cheapest_config(candidate_matrix, slo)
        try:
            got_sel = select_cheapest_config(candidate_matrix, slo)
        except Exception as e:
            out["_note"] = f"Error in select_cheapest_config SLO {slo}: {e}"
            return out

        if got_sel["id"] != want_sel["id"]:
            out["_note"] = f"Selector mismatch on SLO {slo}: got {got_sel['id']}, want {want_sel['id']}"
            return out

    out["replica_matches"] = 1.0
    return out
