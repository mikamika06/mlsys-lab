import importlib

import ref


def check(workdir):
    mod = importlib.import_module("tests.test_convert")
    out = {"flags_expansion": 0.0, "flags_missing_experts": 0.0,
           "clean_silent": 0.0}
    if not hasattr(mod, "audit"):
        return out

    doc = ref.gguf_index("llama")
    warnings = mod.audit(doc["tensors"], doc["metadata"], "llama", experts=0)
    if isinstance(warnings, list) and any(
            "expan" in str(w).lower() or "grow" in str(w).lower() for w in warnings):
        out["flags_expansion"] = 1.0

    moe = ref.gguf_index("qwen3moe")
    w2 = mod.audit(moe["tensors"], moe["metadata"], "qwen3moe", experts=0)
    if isinstance(w2, list) and any("expert" in str(w).lower() for w in w2):
        out["flags_missing_experts"] = 1.0

    # A whole tiny model in F16: every name maps, the shapes agree with the
    # metadata, and nothing is quantised, so there is nothing to report.
    clean = [{"name": "token_embd.weight", "ggml_type": "F16", "ggml_type_id": 1,
              "shape_ggml_order": [16, 8], "n_bytes": 256},
             {"name": "output_norm.weight", "ggml_type": "F16", "ggml_type_id": 1,
              "shape_ggml_order": [16], "n_bytes": 32}]
    for tail in ("attn_q", "attn_k", "attn_v", "attn_output"):
        clean.append({"name": "blk.0.%s.weight" % tail, "ggml_type": "F16",
                      "ggml_type_id": 1, "shape_ggml_order": [16, 16],
                      "n_bytes": 512})
    meta = {"llama.embedding_length": 16, "llama.attention.head_count": 2,
            "llama.attention.head_count_kv": 2, "llama.attention.key_length": 8}
    if mod.audit(clean, meta, "llama", experts=0) == []:
        out["clean_silent"] = 1.0
    return out
