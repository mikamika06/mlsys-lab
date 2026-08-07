from vllmsched.scheduler import Request, Scheduler


def compare_policies_latency(
    requests, num_blocks, block_size, max_tokens, target_req_id
):
    reqs_fcfs = [
        Request(
            r.req_id, r.prompt_len, r.max_gen_len, r.arrival_time, r.priority
        )
        for r in requests
    ]
    reqs_prio = [
        Request(
            r.req_id, r.prompt_len, r.max_gen_len, r.arrival_time, r.priority
        )
        for r in requests
    ]

    sched_fcfs = Scheduler(num_blocks, block_size, max_tokens, policy="fcfs")
    completed_fcfs = sched_fcfs.run_simulation(reqs_fcfs)
    fcfs_req = next(r for r in completed_fcfs if r.req_id == target_req_id)
    fcfs_lat = fcfs_req.completion_time - fcfs_req.arrival_time

    sched_prio = Scheduler(num_blocks, block_size, max_tokens, policy="priority")
    completed_prio = sched_prio.run_simulation(reqs_prio)
    prio_req = next(r for r in completed_prio if r.req_id == target_req_id)
    prio_lat = prio_req.completion_time - prio_req.arrival_time

    return {"fcfs_latency": fcfs_lat, "priority_latency": prio_lat}
