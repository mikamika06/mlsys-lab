import torch
import ref
from lora.module import LoRALinear


def check(workdir):
    torch.manual_seed(1337)
    in_f = 64
    out_f = 32
    r = 8
    alpha = 16.0

    mod = LoRALinear(in_f, out_f, r=r, lora_alpha=alpha, lora_dropout=0.0)
    mod.eval()

    # Create matching reference module by manually setting weights
    ref_mod = torch.nn.Linear(in_f, out_f)
    ref_mod.weight.data.copy_(mod.weight)
    ref_mod.bias.data.copy_(mod.bias)

    with torch.no_grad():
        base_out = ref_mod(torch.randn(2, in_f))
        lora_out = (torch.randn(2, in_f) @ mod.lora_A.T) @ mod.lora_B.T * (alpha / r)
        # Instead, test module directly against manual computation
        x = torch.randn(5, in_f)
        got = mod(x)
        expected = torch.nn.functional.linear(x, mod.weight, mod.bias) + (x @ mod.lora_A.T) @ mod.lora_B.T * (alpha / r)

    diff = torch.max(torch.abs(got - expected)).item()
    match = 1.0 if diff < 1e-6 else 0.0
    return {"output_match": match}
