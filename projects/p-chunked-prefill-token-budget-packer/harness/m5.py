def check(workdir):
    from packer.packer import Packer, Request
    p = Packer(30)
    reqs = [Request(f"r{i}", 60) for i in range(3)]
    for r in reqs:
        p.add_request(r)

    start_step = {}
    finish_step = {}

    for step in range(1, 20):
        alloc = p.step()
        for rid, tokens in alloc.items():
            if rid not in start_step:
                start_step[rid] = step
            r = next(x for x in reqs if x.rid == rid)
            if r.prefill_left == 0 and rid not in finish_step:
                finish_step[rid] = step

    m = {"fifo_packing": 0.0, "all_finished": 0.0}
    if len(finish_step) == 3:
        m["all_finished"] = 1.0
        # Measure how long it took for each request to finish once it started processing.
        # FIFO ensures minimal stretching, RR will cause high stretching.
        stretches = [finish_step[rid] - start_step[rid] + 1 for rid in ["r0", "r1", "r2"]]
        if max(stretches) <= 3:
            m["fifo_packing"] = 1.0
    return m
