import ref
from cbsim.simulator import Request, simulate_continuous

def check(workdir):
    reqs = ref.get_test_workload()
    reqs_ref = [Request(r.req_id, r.arrival_time, r.prompt_len, r.gen_len) for r in reqs]
    reqs_got = [Request(r.req_id, r.arrival_time, r.prompt_len, r.gen_len) for r in reqs]

    ref_comp = simulate_continuous(reqs_ref, max_batch_size=4, max_capacity=128)
    try:
        got_comp = simulate_continuous(reqs_got, max_batch_size=4, max_capacity=128)
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"raised exception: {e}"}

    ref_tokens = sum(r.finish_time for r in ref_comp)
    got_tokens = sum(r.finish_time for r in got_comp)

    if ref_tokens == 0:
        err = 0.0 if got_tokens == 0 else 1.0
    else:
        err = abs(got_tokens - ref_tokens) / ref_tokens

    return {"rel_err": float(err)}
