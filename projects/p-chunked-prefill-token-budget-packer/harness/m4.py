def check(workdir):
    from packer.packer import Packer, Request
    p = Packer(10)
    reqs = [Request(f"r{i}", 20) for i in range(4)]
    for r in reqs:
        p.add_request(r)

    itls = []
    last_step = {}

    for step in range(1, 30):
        alloc = p.step()
        for rid, tokens in alloc.items():
            r = next((x for x in reqs if x.rid == rid), None)
            if r and r.is_decode and tokens == 1:
                if rid in last_step:
                    itls.append(step - last_step[rid])
                last_step[rid] = step

    m = {"itl_below_threshold": 0.0, "has_decodes": 0.0}
    if itls:
        m["has_decodes"] = 1.0
        # If progress is guaranteed, ITL should perfectly be 1 (scheduled every step)
        if max(itls) <= 1:
            m["itl_below_threshold"] = 1.0
    return m
