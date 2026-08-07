def locate_boundary_memcpys(profile):
    res = []
    for n in profile["nodes"]:
        op = n["op_type"]
        if "Memcpy" in op:
            res.append(n)
    return res
