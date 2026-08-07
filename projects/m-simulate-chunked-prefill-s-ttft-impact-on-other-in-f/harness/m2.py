import sys
import ref

def check(workdir):
    if workdir not in sys.path:
        sys.path.insert(0, workdir)
    try:
        from llm_sched.scheduler import simulate_schedule

        ttft_err, stall_err, total_err = 0.0, 0.0, 0.0

        for sc in ref.SCENARIOS:
            want = ref.simulate_schedule(sc["prompt"], sc["reqs"], sc["chunk"], sc["p_cost"], sc["d_cost"])
            got = simulate_schedule(sc["prompt"], sc["reqs"], sc["chunk"], sc["p_cost"], sc["d_cost"])

            ttft_err = max(ttft_err, abs(want["ttft"] - got["ttft"]) / (want["ttft"] + 1e-9))
            stall_err = max(stall_err, abs(want["max_stall"] - got["max_stall"]) / (want["max_stall"] + 1e-9))
            total_err = max(total_err, abs(want["total_time"] - got["total_time"]) / (want["total_time"] + 1e-9))

        return {"ttft_rel_err": ttft_err, "stall_rel_err": stall_err, "total_rel_err": total_err}
    except Exception as e:
        return {"_note": str(e), "ttft_rel_err": 1.0, "stall_rel_err": 1.0, "total_rel_err": 1.0}
