import ref


def check(workdir):
    from forensics.simulator import simulate_server as learner_sim

    requests = [{"tokens": 128}, {"tokens": 256}, {"tokens": 64}, {"tokens": 512}, {"tokens": 128}]
    capacity = 16
    max_seqs = 2

    ref_res = ref.simulate_server(requests, capacity, max_seqs)
    try:
        got_res = learner_sim(requests, capacity, max_seqs)
    except Exception as e:
        return {"preemption_count_match": 0.0, "queue_delay_match": 0.0, "_note": f"raised {e}"}

    out = {"preemption_count_match": 0.0, "queue_delay_match": 0.0}
    if got_res is not None:
        if got_res.get("preemption_count") == ref_res["preemption_count"]:
            out["preemption_count_match"] = 1.0
        if abs(got_res.get("total_queue_delay", -1) - ref_res["total_queue_delay"]) < 1e-3:
            out["queue_delay_match"] = 1.0
    return out
