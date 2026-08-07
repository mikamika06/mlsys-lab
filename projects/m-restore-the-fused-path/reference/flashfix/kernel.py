def audit_contiguity(configs):
    results = []
    for cfg in configs:
        shape = cfg["shape"]
        strides = cfg["strides"]
        expected_stride = 1
        is_contig = True
        for dim, stride in zip(reversed(shape), reversed(strides)):
            if stride != expected_stride:
                is_contig = False
                break
            expected_stride *= dim
        results.append({"layer_id": cfg["layer_id"], "contiguous": is_contig and cfg["contiguous"]})
    return results
