def compute_layer_flops(config, seq_len):
    raise NotImplementedError

def compute_attention_flops(config, seq_len, causal=True):
    raise NotImplementedError

def compute_total_flops(config, prefill_len, decode_steps):
    raise NotImplementedError

def compute_mfu(config, measured_time, total_flops, peak_tflops):
    raise NotImplementedError

class MFUCalculator:
    def __init__(self, config):
        raise NotImplementedError
    def evaluate(self, workload):
        raise NotImplementedError
