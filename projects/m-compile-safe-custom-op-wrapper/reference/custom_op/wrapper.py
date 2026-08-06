import torch

@torch.library.custom_op("custom_flash::flash_attn", mutates_args=())
def flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = 1.0 / (q.shape[-1] ** 0.5)
    attn = torch.matmul(q, k.transpose(-1, -2)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)

@flash_attn.register_fake
def flash_attn_fake(q, k, v):
    return torch.empty_like(q)

def run_attention(q, k, v):
    return flash_attn(q, k, v)
