import torch
import torch.nn.functional as F
import math

def compute_manual(q, k, v, pad_mask):
    B, H, S, D = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    mask = pad_mask.view(B, 1, 1, -1)
    scores = scores.masked_fill(mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)

def compute_sdpa(q, k, v, pad_mask):
    B, H, S, D = q.shape
    mask = pad_mask.view(B, 1, 1, -1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=~mask)
