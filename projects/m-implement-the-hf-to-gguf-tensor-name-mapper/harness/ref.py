import re
import numpy as np

MAPPINGS = [
    (r"^model\.embed_tokens\.weight$", "token_embd.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$", r"blk.\1.attn_q.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$", r"blk.\1.attn_k.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$", r"blk.\1.attn_v.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$", r"blk.\1.attn_output.\2"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.weight$", r"blk.\1.attn_norm.weight"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$", r"blk.\1.ffn_norm.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$", r"blk.\1.ffn_gate.\2"),
    (r"^model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$", r"blk.\1.ffn_up.\2"),
    (r"^model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$", r"blk.\1.ffn_down.\2"),
    (r"^model\.norm\.weight$", "output_norm.weight"),
    (r"^lm_head\.weight$", "output.weight"),
]


def map_hf_to_gguf(hf_name: str) -> str:
    for pattern, repl in MAPPINGS:
        if re.match(pattern, hf_name):
            return re.sub(pattern, repl, hf_name)
    raise ValueError(f"Unknown HF tensor name: {hf_name}")


def undo_rope_permutation(w: np.ndarray, n_heads: int) -> np.ndarray:
    shape = w.shape
    total_dim = shape[0]
    head_dim = total_dim // n_heads
    w_3d = w.reshape(n_heads, head_dim, -1)
    half_dim = head_dim // 2
    out_3d = np.zeros_like(w_3d)
    out_3d[:, 0::2, :] = w_3d[:, :half_dim, :]
    out_3d[:, 1::2, :] = w_3d[:, half_dim:, :]
    return out_3d.reshape(shape)


def match_tensors(tensor_names: list[str], pattern: str) -> list[str]:
    regex = re.compile(pattern)
    return [name for name in tensor_names if regex.search(name)]


HF_NAMES = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.q_proj.bias",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.15.self_attn.q_proj.weight",
    "model.layers.15.mlp.down_proj.bias",
    "model.norm.weight",
    "lm_head.weight",
]


def _gen_rope_fixtures():
    fixtures = []
    configs = [(4, 16, 32), (8, 64, 128), (2, 8, 1)]
    for n_heads, head_dim, hidden in configs:
        tot = n_heads * head_dim
        if hidden == 1:
            arr = np.arange(tot, dtype=np.float32)
        else:
            arr = np.arange(tot * hidden, dtype=np.float32).reshape(tot, hidden)
        fixtures.append({"tensor": arr, "n_heads": n_heads})
    return fixtures


ROPE_TEST_CASES = _gen_rope_fixtures()
