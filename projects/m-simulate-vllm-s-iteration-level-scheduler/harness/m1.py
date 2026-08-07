import ref


def check(workdir):
    from vllmsched.scheduler import Request, Scheduler

    out = {"scheduler_match": 0.0}

    reqs_ref = [
        ref.Request(
            req_id=1, prompt_len=16, max_gen_len=10, arrival_time=0, priority=1
        ),
        ref.Request(
            req_id=2, prompt_len=32, max_gen_len=5, arrival_time=0, priority=2
        ),
        ref.Request(
            req_id=3, prompt_len=8, max_gen_len=12, arrival_time=1, priority=5
        ),
    ]
    sched_ref = ref.Scheduler(
        num_blocks=8, block_size=16, max_num_batched_tokens=40, policy="priority"
    )
    res_ref = sched_ref.run_simulation(reqs_ref)

    reqs_user = [
        Request(
            req_id=1, prompt_len=16, max_gen_len=10, arrival_time=0, priority=1
        ),
        Request(
            req_id=2, prompt_len=32, max_gen_len=5, arrival_time=0, priority=2
        ),
        Request(
            req_id=3, prompt_len=8, max_gen_len=12, arrival_time=1, priority=5
        ),
    ]
    sched_user = Scheduler(
        num_blocks=8, block_size=16, max_num_batched_tokens=40, policy="priority"
    )
    res_user = sched_user.run_simulation(reqs_user)

    ref_map = {
        r.req_id: (r.scheduled_time, r.completion_time) for r in res_ref
    }
    user_map = {
        r.req_id: (r.scheduled_time, r.completion_time) for r in res_user
    }

    if ref_map == user_map:
        out["scheduler_match"] = 1.0
    else:
        out["_note"] = f"Expected schedule timings {ref_map}, got {user_map}"

    return out
