def check(workdir):
    from packer.packer import Packer, Request
    p = Packer(100)
    r1 = Request("r1", 50)
    r2 = Request("r2", 200)
    p.add_request(r1)

    # Process r1 to completion of prefill
    p.step()

    # Add a large prefill request
    p.add_request(r2)
    s2 = p.step()

    m = {"decode_progress": 0.0, "prefill_progress": 0.0}
    if s2.get("r1") == 1:
        m["decode_progress"] = 1.0
    if s2.get("r2") == 99:
        m["prefill_progress"] = 1.0
    return m
