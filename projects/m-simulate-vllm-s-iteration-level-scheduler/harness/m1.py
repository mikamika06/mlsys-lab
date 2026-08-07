import ref

def check(workdir):
    from vllmsched.scheduler import simulate_scheduler, Request
    suites = ref.get_test_suites()
    matched = 0
    for suite in suites:
        reqs_ref = [ref.RequestRef(r["req_id"], r["prompt_len"], r["gen_len"], r.get("priority", 0)) for r in suite]
        want = ref.simulate_scheduler(reqs_ref, policy="fcfs", max_num_seqs=4)
        reqs_learner = [Request(r["req_id"], r["prompt_len"], r["gen_len"], r.get("priority", 0)) for r in suite]
        try:
            got = simulate_scheduler(reqs_learner, policy="fcfs", max_num_seqs=4)
            if got == want:
                matched += 1
        except Exception:
            pass
    out = {"schedule_matched": 1.0 if matched == len(suites) else 0.0}
    if matched < len(suites):
        out["_note"] = f"matched {matched}/{len(suites)} scheduler suites"
    return out
