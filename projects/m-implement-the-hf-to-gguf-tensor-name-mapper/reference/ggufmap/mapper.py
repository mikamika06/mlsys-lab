import re

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
    """Map a HuggingFace tensor name to GGUF format."""
    for pattern, repl in MAPPINGS:
        if re.match(pattern, hf_name):
            return re.sub(pattern, repl, hf_name)
    raise ValueError(f"Unknown HF tensor name: {hf_name}")
