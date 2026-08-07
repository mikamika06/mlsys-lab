def check(workdir):
    from packer.packer import Packer, Request
    p = Packer(100)
    r1 = Request("r1", 60)
    r2 = Request("r2", 80)
    p.add_request(r1)
    p.add_request(r2)

    s1 = p.step()
    s2 = p.step()

    m = {"packed_first": 0.0, "budget_respected": 0.0}
    if s1.get("r1") == 60 and s1.get("r2") == 40:
        m["packed_first"] = 1.0
    if sum(s1.values()) <= 100 and sum(s2.values()) <= 100:
        m["budget_respected"] = 1.0
    return m
