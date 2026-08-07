import numpy as np


class NaiveMHA:

    def __init__(self, embed_dim: int, num_heads: int, qkv_w: np.ndarray,
        out_w: np.ndarray):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_w = qkv_w
        self.out_w = out_w

    def forward(self, x: np.ndarray) ->np.ndarray:
        B, S, E = x.shape
        qkv = x @ self.qkv_w
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        max_s = np.max(attn_scores, axis=-1, keepdims=True)
        exp_s = np.exp(attn_scores - max_s)
        attn_probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        out = attn_probs @ v
        out = out.transpose(0, 2, 1, 3).reshape(B, S, E)
        return out @ self.out_w


class ANEFriendlyMHA:

    def __init__(self, embed_dim: int, num_heads: int, qkv_w: np.ndarray,
        out_w: np.ndarray):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_w = qkv_w
        self.out_w = out_w

    def forward(self, x: np.ndarray) ->np.ndarray:
        B, C, H_dim, S = x.shape
        qkv_w_conv = self.qkv_w.T[:, :, None, None]
        qkv = np.zeros((B, 3 * C, 1, S), dtype=x.dtype)
        for oc in range(3 * C):
            for ic in range(C):
                qkv[:, oc, 0, :] += qkv_w_conv[oc, ic, 0, 0] * x[:, ic, 0, :]
        q = qkv[:, :C, 0, :]
        k = qkv[:, C:2 * C, 0, :]
        v = qkv[:, 2 * C:, 0, :]
        q_h = q.reshape(B, self.num_heads, self.head_dim, S)
        k_h = k.reshape(B, self.num_heads, self.head_dim, S)
        v_h = v.reshape(B, self.num_heads, self.head_dim, S)
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.einsum('bhds,bhdk->bhsk', q_h, k_h) * scale
        max_s = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - max_s)
        attn_probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        ctx_h = np.einsum('bhds,bhks->bhkd', v_h, attn_probs)
        ctx_h = ctx_h.transpose(0, 1, 3, 2)
        ctx_bc1s = ctx_h.reshape(B, C, 1, S)
        out_w_conv = self.out_w.T[:, :, None, None]
        out = np.zeros((B, C, 1, S), dtype=x.dtype)
        for oc in range(C):
            for ic in range(C):
                out[:, oc, 0, :] += out_w_conv[oc, ic, 0, 0] * ctx_bc1s[:, ic, 0, :]
        return out
