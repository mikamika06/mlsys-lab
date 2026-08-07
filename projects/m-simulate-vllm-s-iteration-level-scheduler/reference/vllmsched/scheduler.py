class Request:
    def __init__(self, req_id, prompt_len, gen_len, priority=0):
        self.req_id = req_id
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.priority = priority
        self.tokens_generated = 0
        self.status = "waiting"

def simulate_scheduler(requests, policy="fcfs", max_num_seqs=4):
    waiting = list(requests)
    running = []
    completed = []
    time_step = 0
    while waiting or running:
        if policy == "fcfs":
            waiting.sort(key=lambda r: r.req_id)
        elif policy == "priority":
            waiting.sort(key=lambda r: (-r.priority, r.req_id))
        while waiting and len(running) < max_num_seqs:
            req = waiting.pop(0)
            req.status = "running"
            running.append(req)
        for req in running:
            req.tokens_generated += 1
            if req.tokens_generated >= req.gen_len:
                req.status = "completed"
        finished = [r for r in running if r.status == "completed"]
        for r in finished:
            running.remove(r)
            completed.append(r)
        time_step += 1
        if time_step > 10000:
            break
    return [{"req_id": r.req_id, "gen_len": r.gen_len, "priority": r.priority} for r in completed]

def compare_policies(requests, target_req_id):
    reqs_fcfs = [Request(r["req_id"], r["prompt_len"], r["gen_len"], r.get("priority", 0)) for r in requests]
    reqs_prio = [Request(r["req_id"], r["prompt_len"], r["gen_len"], r.get("priority", 0)) for r in requests]
    res_fcfs = simulate_scheduler(reqs_fcfs, policy="fcfs")
    res_prio = simulate_scheduler(reqs_prio, policy="priority")
    return {"fcfs_order": [r["req_id"] for r in res_fcfs], "priority_order": [r["req_id"] for r in res_prio]}
