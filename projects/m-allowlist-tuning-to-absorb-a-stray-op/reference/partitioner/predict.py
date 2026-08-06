def partition_ops(ops, allowlist):
    res = []
    current_blob = -1
    in_blob = False
    for op in ops:
        if op['type'] in allowlist:
            if not in_blob:
                current_blob += 1
                in_blob = True
            res.append(current_blob)
        else:
            in_blob = False
            res.append(-1)
    return res
