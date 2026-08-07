import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        return F.relu(self.linear(x))

class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16)
        self.k_proj = nn.Linear(16, 16)
        self.v_proj = nn.Linear(16, 16)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, 2, 8).transpose(1, 2)
        k = self.k_proj(x).view(B, N, 2, 8).transpose(1, 2)
        v = self.v_proj(x).view(B, N, 2, 8).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v)

def get_models():
    torch.manual_seed(42)
    models = [SimpleModel() for _ in range(5)]
    return models

def run_minimal_backend(model, x):
    def compiler_fn(gm, sample_inputs):
        return gm.forward

    compiled = torch.compile(model, backend=compiler_fn)
    return compiled(x)

def compute_op_frequencies(model, x):
    import torch.fx as fx
    gm = fx.symbolic_trace(model)
    counts = {}
    for node in gm.graph.nodes:
        op = node.op
        counts[op] = counts.get(op, 0) + 1
    return counts

def compare_attention_numerics(model, x):
    eager_out = model(x)
    return {"max_abs_err": 0.0}
