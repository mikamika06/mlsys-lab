def analytic_latency(batch_meta, hardware_specs):
    bs = batch_meta["batch_size"]
    max_len = batch_meta["max_seqlen"]
    bw = hardware_specs["bandwidth"]
    flop = hardware_specs["flops"]
    ttft = (max_len * bs * 1024) / flop + 0.001
    itl = (bs * 512) / bw + 0.0005
    return {"ttft": float(ttft), "itl": float(itl)}
