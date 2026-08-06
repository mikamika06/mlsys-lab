import torch
import math
import torch.nn.functional as F

def get_fixture():
    torch.manual_seed(42)
    B, H, S, D = 2, 4, 8, 16
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    pad_mask = torch.zeros(B, S, dtype=torch.bool)
    pad_mask[0, 5:] = True
    pad_mask[1, 7:] = True
    return q, k, v, pad_mask

def get_expected():
    q, k, v, pad_mask = get_fixture()
    B, H, S, D = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
    mask = pad_mask.view(B, 1, 1, -1)
    scores = scores.masked_fill(mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, v)
