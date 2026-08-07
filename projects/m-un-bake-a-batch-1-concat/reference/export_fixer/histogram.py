def op_histogram_diff(dynamo_ops, ts_ops):
    all_keys = set(dynamo_ops.keys()) | set(ts_ops.keys())
    diff = {}
    for k in all_keys:
        diff[k] = abs(dynamo_ops.get(k, 0) - ts_ops.get(k, 0))
    return diff
