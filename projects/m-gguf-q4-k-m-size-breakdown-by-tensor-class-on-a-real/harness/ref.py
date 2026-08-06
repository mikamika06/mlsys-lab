import random

def generate_fixtures():
    random.seed(42)
    tensors = []
    for i in range(4):
        tensors.append((f"blk.{i}.attn_q.weight", 1048576, "Q4_K_M"))
        tensors.append((f"blk.{i}.router.weight", 262144, "Q4_K_M"))
        for e in range(8):
            tensors.append((f"blk.{i}.ffn_experts.{e}.weight", 2097152, "Q4_K_M"))
    return tensors

TENSORS = generate_fixtures()
ACTIVATION_TRACES = {e: [random.uniform(0.1, 15.0) for _ in range(10)] for e in range(32)}
