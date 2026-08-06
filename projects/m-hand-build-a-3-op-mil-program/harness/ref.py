def generate_program_oracle():
    return [
        {"name": "transpose", "inputs": ["x"], "outputs": ["x_t"], "attributes": {"perm": [0, 2, 1, 3]}},
        {"name": "matmul", "inputs": ["x_t", "weights"], "outputs": ["attn_score"], "attributes": {}},
        {"name": "softmax", "inputs": ["attn_score"], "outputs": ["attn_out"], "attributes": {"axis": -1}}
    ]

def generate_pass_oracle(seq_len):
    base_ops = 3 * seq_len
    fused_ops = int(seq_len * 0.4) + 2
    memory_unfused = seq_len * 1024
    memory_fused = int(seq_len * 512)
    return {
        "ops_before": base_ops,
        "ops_after": fused_ops,
        "memory_bytes_before": memory_unfused,
        "memory_bytes_after": memory_fused,
        "fused": True
    }

SAMPLE_DUMP = """
MIL Text Dump v1.0
------------------
  transpose : 12
  matmul : 8
  softmax : 4
  elementwise_add : 20
"""

SAMPLE_HISTOGRAM = {
    "transpose": 12,
    "matmul": 8,
    "softmax": 4,
    "elementwise_add": 20
}
