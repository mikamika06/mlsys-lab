import numpy as np

def compute_layer_flops(config, seq_len):
    hidden_size = config["hidden_size"]
    num_heads = config["num_heads"]
    num_kv_heads = config.get("num_kv_heads", num_heads)
    intermediate_size = config["intermediate_size"]
    head_dim = hidden_size // num_heads

    q_flops = 2 * seq_len * hidden_size * (num_heads * head_dim)
    k_flops = 2 * seq_len * hidden_size * (num_kv_heads * head_dim)
    v_flops = 2 * seq_len * hidden_size * (num_kv_heads * head_dim)
    o_flops = 2 * seq_len * (num_heads * head_dim) * hidden_size
    attn_proj_flops = q_flops + k_flops + v_flops + o_flops

    attn_score_flops = 2 * num_heads * seq_len * seq_len * head_dim

    mlp_gate_flops = 2 * seq_len * hidden_size * intermediate_size
    mlp_up_flops = 2 * seq_len * hidden_size * intermediate_size
    mlp_down_flops = 2 * seq_len * intermediate_size * hidden_size
    mlp_flops = mlp_gate_flops + mlp_up_flops + mlp_down_flops

    return attn_proj_flops + attn_score_flops + mlp_flops

def compute_attention_flops(config, seq_len, causal=True):
    num_heads = config["num_heads"]
    hidden_size = config["hidden_size"]
    head_dim = hidden_size // num_heads
    factor = 0.5 if causal else 1.0
    return 2 * num_heads * seq_len * seq_len * head_dim * factor

def compute_total_flops(config, prefill_len, decode_steps):
    num_layers = config["num_layers"]
    layer_prefill = compute_layer_flops(config, prefill_len)
    total_prefill = num_layers * layer_prefill

    decode_total = 0.0
    for step in range(decode_steps):
        current_seq = prefill_len + step
        decode_total += num_layers * compute_layer_flops(config, 1)
        decode_total += num_layers * 2 * config["num_heads"] * current_seq * (config["hidden_size"] // config["num_heads"])

    return total_prefill + decode_total

def compute_mfu(config, measured_time, total_flops, peak_tflops):
    if measured_time <= 0:
        return 0.0
    achieved_tflops = (total_flops / measured_time) / 1e12
    return float(achieved_tflops / peak_tflops)

class MFUCalculator:
    def __init__(self, config):
        self.config = config

    def evaluate(self, workload):
        prefill_len = workload["prefill_len"]
        decode_steps = workload["decode_steps"]
        measured_time = workload["measured_time"]
        peak_tflops = workload["peak_tflops"]

        total = compute_total_flops(self.config, prefill_len, decode_steps)
        return compute_mfu(self.config, measured_time, total, peak_tflops)
