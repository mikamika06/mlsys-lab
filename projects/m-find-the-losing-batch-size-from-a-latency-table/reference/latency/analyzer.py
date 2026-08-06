def find_losing_batch_size(table, slo_latency):
    losing_bs = None
    for row in table:
        if row["latency"] > slo_latency:
            losing_bs = row["batch_size"]
            break
    if losing_bs is None and table:
        losing_bs = table[-1]["batch_size"]
    return losing_bs
