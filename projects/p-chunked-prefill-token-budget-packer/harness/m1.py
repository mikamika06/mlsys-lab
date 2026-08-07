def check(workdir):
    from packer.packer import Packer, Request
    p = Packer(100)
    r1 = Request("r1", 250)
    p.add_request(r1)

    s1 = p.step()
    s2 = p.step()
    s3 = p.step()

    m = {"split_chunks": 0.0, "finished": 0.0}
    if s1.get("r1") == 100 and s2.get("r1") == 100 and s3.get("r1") == 50:
        m["split_chunks"] = 1.0
    if r1.prefill_left == 0 and r1.is_decode:
        m["finished"] = 1.0
    return m
