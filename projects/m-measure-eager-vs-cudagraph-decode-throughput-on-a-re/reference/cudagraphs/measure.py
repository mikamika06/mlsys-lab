def simulate_eager_throughput(config):
    bs = config["batch_size"]
    seq = config["seq_len"]
    return float(10000.0 / (bs * 1.5 + seq * 0.01))
