import ref
from cbsim.simulator import Request, compute_throughput_ratio

def check(workdir):
    reqs = ref.get_test_workload()
    reqs_ref = [Request(r.req_id, r.arrival_time, r.prompt_len, r.gen_len) for r in reqs]
    reqs_got = [Request(r.req_id, r.arrival_time, r.prompt_len, r.gen_len) for r in reqs]

    ref_ratio = compute_throughput_ratio(reqs_ref, max_batch_size=4, max_capacity=128)
    try:
        got_ratio = compute_throughput_ratio(reqs_got, max_batch_size=4, max_capacity=128)
    except Exception as e:
        return {"rel_err": 1.0, "_note": f"raised exception: {e}"}

    if ref_ratio == 0:
        err = 0.0 if got_ratio == 0 else 1.0
    else:
        err = abs(got_ratio - ref_ratio) / ref_ratio

    return {"rel_err": float(err)}
