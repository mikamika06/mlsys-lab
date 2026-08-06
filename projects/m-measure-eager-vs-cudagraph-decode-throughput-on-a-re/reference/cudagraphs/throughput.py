import reference.cudagraphs.measure as m

def simulate_cudagraph_throughput(config):
    bs = config["batch_size"]
    seq = config["seq_len"]
    return float(12000.0 / (bs * 1.2 + seq * 0.008))

def calculate_throughput_ratio(config):
    eager = m.simulate_eager_throughput(config)
    cg = simulate_cudagraph_throughput(config)
    return cg / eager
