#!/usr/bin/env python3
"""Fixtures for the checkpoint-conversion units.

Converting a checkpoint is mostly a naming and layout problem, and the names
are the part that cannot be invented: which tensors a real llama export calls
`blk.0.attn_q.weight`, which ones a mixture-of-experts model adds, which ones
have no counterpart on the other side at all. So the tensor indexes here are
lifted whole from real models on this machine — every name, shape and ggml
type, without the weights.

The safetensors side is written by the reference writer and then sharded the
way a real export is, index file included, because the shard index is where a
hand-written loader gets the offsets wrong.

    python3 tools/record_convert.py
"""
import json
import os
import struct
import sys

import numpy as np

try:
    import gguf
    from gguf import GGUFReader
except ImportError:
    sys.exit("needs the `gguf` package")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "projects", "_fixtures")
BLOBS = os.path.expanduser("~/.ollama/models/blobs")


def tensor_index(path, out_name):
    r = GGUFReader(path)
    arch = r.fields.get("general.architecture")
    rows = []
    for t in r.tensors:
        qt = gguf.GGMLQuantizationType(int(t.tensor_type))
        rows.append({"name": t.name, "ggml_type": qt.name,
                     "ggml_type_id": int(t.tensor_type),
                     "shape_ggml_order": [int(x) for x in t.shape.tolist()],
                     "n_bytes": int(t.n_bytes)})
    meta = {}
    for key in ("general.architecture", "general.name", "llama.block_count",
                "llama.attention.head_count", "llama.attention.head_count_kv",
                "llama.embedding_length", "llama.feed_forward_length",
                "llama.attention.key_length", "llama.attention.value_length",
                "llama.context_length", "llama.rope.dimension_count",
                "qwen3moe.block_count", "qwen3moe.expert_count",
                "qwen3moe.expert_used_count", "qwen3moe.embedding_length",
                "qwen3moe.attention.head_count", "qwen3moe.attention.head_count_kv",
                "qwen3moe.attention.key_length", "qwen3moe.attention.value_length",
                "qwen3moe.feed_forward_length", "qwen3moe.expert_feed_forward_length"):
        f = r.fields.get(key)
        if f is not None:
            try:
                v = f.contents()
                meta[key] = v.decode() if isinstance(v, bytes) else (
                    v.item() if isinstance(v, np.generic) else v)
            except Exception:
                pass
    doc = {"architecture": arch.contents() if arch else "?",
           "tensor_count": len(rows), "metadata": meta, "tensors": rows}
    with open(os.path.join(FIX, "gguf", out_name), "w") as f:
        json.dump(doc, f, indent=2)
    return doc


def shard_safetensors():
    """A two-shard export with an index, mixed dtypes, and a damaged copy."""
    from safetensors.numpy import save_file

    out = os.path.join(FIX, "safetensors")
    os.makedirs(out, exist_ok=True)
    src = np.load(os.path.join(FIX, "gguf", "dequantized_truth.npz"))
    # A conversion planner reads names, shapes and offsets, not values, so a few
    # rows of each tensor carry the same lesson at a twentieth of the size.
    base = {k: (src[k][:8] if src[k].ndim > 1 else src[k][:512]) for k in src.files}

    plan = {}
    for i, (name, arr) in enumerate(sorted(base.items())):
        stem = name.replace(".", "_")
        plan[stem + ".f32"] = arr.astype(np.float32)
        plan[stem + ".f16"] = arr.astype(np.float16)
        if i == 0:
            # bf16 has no numpy scalar type, so a hand-written parser has to
            # know its width from the header rather than from a dtype table.
            raw = arr.astype(np.float32).view(np.uint32) >> 16
            plan[stem + ".bf16"] = raw.astype(np.uint16)

    names = sorted(plan)
    half = len(names) // 2 or 1
    shards = {"model-00001-of-00002.safetensors": names[:half],
              "model-00002-of-00002.safetensors": names[half:]}
    weight_map, total = {}, 0
    for fname, keys in shards.items():
        save_file({k: plan[k] for k in keys}, os.path.join(out, fname),
                  metadata={"format": "np"})
        total += os.path.getsize(os.path.join(out, fname))
        for k in keys:
            weight_map[k] = fname
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    with open(os.path.join(out, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    truth = {}
    for fname in shards:
        p = os.path.join(out, fname)
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
        truth[fname] = {"header_bytes": n, "file_bytes": os.path.getsize(p),
                        "header": head}
    with open(os.path.join(out, "header_truth.json"), "w") as f:
        json.dump(truth, f, indent=2)

    # One shard with a tensor whose end offset runs past the file.
    good = os.path.join(out, "model-00002-of-00002.safetensors")
    with open(good, "rb") as f:
        blob = bytearray(f.read())
    n = struct.unpack("<Q", bytes(blob[:8]))[0]
    head = json.loads(bytes(blob[8:8 + n]))
    victim = sorted(k for k in head if k != "__metadata__")[-1]
    head[victim]["data_offsets"][1] += 4096
    new = json.dumps(head, separators=(",", ":")).encode()
    new = new + b" " * ((8 - len(new) % 8) % 8)
    damaged = bytearray(struct.pack("<Q", len(new))) + new + blob[8 + n:]
    with open(os.path.join(out, "model-damaged.safetensors"), "wb") as f:
        f.write(bytes(damaged))
    with open(os.path.join(out, "damage_truth.json"), "w") as f:
        json.dump({"file": "model-damaged.safetensors", "tensor": victim,
                   "field": "data_offsets[1]", "shift_bytes": 4096,
                   "why": "declared end runs past the end of the file"}, f, indent=2)
    return {"shards": len(shards), "tensors": len(plan), "damaged_tensor": victim}


def mlx_tree():
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError:
        return {"skipped": "mlx not installed"}

    class Attn(nn.Module):
        def __init__(self, d, heads, kv_heads):
            super().__init__()
            head = d // heads
            self.q_proj = nn.Linear(d, heads * head, bias=False)
            self.k_proj = nn.Linear(d, kv_heads * head, bias=False)
            self.v_proj = nn.Linear(d, kv_heads * head, bias=False)
            self.o_proj = nn.Linear(heads * head, d, bias=False)

    class MLP(nn.Module):
        def __init__(self, d, ff):
            super().__init__()
            self.gate_proj = nn.Linear(d, ff, bias=False)
            self.up_proj = nn.Linear(d, ff, bias=False)
            self.down_proj = nn.Linear(ff, d, bias=False)

    class Layer(nn.Module):
        def __init__(self, d, heads, kv_heads, ff):
            super().__init__()
            self.self_attn = Attn(d, heads, kv_heads)
            self.mlp = MLP(d, ff)
            self.input_layernorm = nn.RMSNorm(d)
            self.post_attention_layernorm = nn.RMSNorm(d)

    class Model(nn.Module):
        def __init__(self, n, d, heads, kv_heads, ff, vocab):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, d)
            self.layers = [Layer(d, heads, kv_heads, ff) for _ in range(n)]
            self.norm = nn.RMSNorm(d)
            self.lm_head = nn.Linear(d, vocab, bias=False)

    m = Model(2, 256, 8, 2, 512, 1000)
    flat = {}

    def walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(f"{prefix}.{i}", v)
        else:
            flat[prefix] = [int(x) for x in obj.shape]

    walk("", m.parameters())
    doc = {"module": "mlx.nn llama-shaped model, 2 layers",
           "mlx_version": mx.__version__,
           "config": {"layers": 2, "hidden": 256, "heads": 8, "kv_heads": 2,
                      "ffn": 512, "vocab": 1000},
           "params": dict(sorted(flat.items()))}
    with open(os.path.join(FIX, "mlx", "mlx_param_tree.json"), "w") as f:
        json.dump(doc, f, indent=2)
    return {"params": len(flat)}


def main():
    report = {}
    pairs = [("sha256-b3a2c9a8fef9be8d2ef951aecca36a36b9ea0b70abe9359eab4315bf4cd9be01",
              "tensor_index_llama.json"),
             ("sha256-1194192cf2a187eb02722edcc3f77b11d21f537048ce04b67ccf8ba78863006a",
              "tensor_index_qwen3moe.json")]
    for blob, name in pairs:
        p = os.path.join(BLOBS, blob)
        if not os.path.isfile(p):
            report[name] = {"skipped": "blob missing"}
            continue
        doc = tensor_index(p, name)
        report[name] = {"arch": doc["architecture"], "tensors": doc["tensor_count"]}
    report["safetensors"] = shard_safetensors()
    report["mlx"] = mlx_tree()
    for k, v in report.items():
        print(f"{k:32} {json.dumps(v)[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
