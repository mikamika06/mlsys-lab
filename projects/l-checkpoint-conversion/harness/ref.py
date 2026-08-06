import json
import os
import struct

FIX = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "_fixtures"))
ST = os.path.join(FIX, "safetensors")
GG = os.path.join(FIX, "gguf")
MLX = os.path.join(FIX, "mlx")

WIDTH = {"F32": 4, "F16": 2, "BF16": 2, "U16": 2, "I64": 8, "U8": 1, "BOOL": 1}


def blob(name):
    with open(os.path.join(ST, name), "rb") as f:
        return f.read()


def shard_names():
    return ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


def header_truth():
    with open(os.path.join(ST, "header_truth.json"), encoding="utf-8") as f:
        return json.load(f)


def damage_truth():
    with open(os.path.join(ST, "damage_truth.json"), encoding="utf-8") as f:
        return json.load(f)


def index_path():
    return os.path.join(ST, "model.safetensors.index.json")


def gguf_index(which):
    with open(os.path.join(GG, "tensor_index_%s.json" % which), encoding="utf-8") as f:
        return json.load(f)


def mlx_params():
    with open(os.path.join(MLX, "mlx_param_tree.json"), encoding="utf-8") as f:
        return json.load(f)["params"]


def expect_entries(name):
    body = blob(name)
    n = struct.unpack_from("<Q", body, 0)[0]
    head = json.loads(body[8:8 + n])
    out = []
    for key, rec in head.items():
        if key == "__metadata__":
            continue
        elems = 1
        for d in rec["shape"]:
            elems *= int(d)
        start, end = (int(x) for x in rec["data_offsets"])
        out.append({"name": key, "dtype": rec["dtype"],
                    "shape": [int(x) for x in rec["shape"]],
                    "elements": elems,
                    "declared_bytes": end - start,
                    "expected_bytes": elems * WIDTH.get(rec["dtype"], 0),
                    "absolute_offsets": [8 + n + start, 8 + n + end]})
    out.sort(key=lambda e: e["absolute_offsets"][0])
    return {"data_start": 8 + n, "tensors": out}


def expect_map(which):
    """Independent name mapping, written from the rule table in the brief."""
    doc = gguf_index(which)
    experts = doc["metadata"].get("%s.expert_count" % doc["architecture"], 0)
    per = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
           "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
           "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm",
           "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj",
           "ffn_down": "mlp.down_proj", "attn_q_norm": "self_attn.q_norm",
           "attn_k_norm": "self_attn.k_norm", "ffn_gate_inp": "mlp.gate"}
    exps = {"ffn_gate_exps": "mlp.experts.%d.gate_proj",
            "ffn_up_exps": "mlp.experts.%d.up_proj",
            "ffn_down_exps": "mlp.experts.%d.down_proj"}
    top = {"token_embd.weight": "embed_tokens.weight",
           "output_norm.weight": "norm.weight", "output.weight": "lm_head.weight"}
    mapped, fanned, unmapped = {}, {}, []
    for t in doc["tensors"]:
        name = t["name"]
        if name in top:
            mapped[name] = top[name]
            continue
        parts = name.split(".")
        if len(parts) < 4 or parts[0] != "blk":
            unmapped.append(name)
            continue
        layer, tail = int(parts[1]), parts[2]
        if tail in per:
            mapped[name] = "layers.%d.%s.%s" % (layer, per[tail], parts[3])
        elif tail in exps and experts:
            fanned[name] = ["layers.%d.%s.%s" % (layer, exps[tail] % e, parts[3])
                            for e in range(experts)]
        else:
            unmapped.append(name)
    return {"mapped": mapped, "fanned_out": fanned, "unmapped": unmapped,
            "target_count": len(mapped) + sum(len(v) for v in fanned.values()),
            "experts": experts, "doc": doc}


def expect_plan(which, out_dtype="F16"):
    doc = gguf_index(which)
    width = {"F16": 2, "BF16": 2, "F32": 4}[out_dtype]
    quant = {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K",
             "Q5_K", "Q6_K", "Q8_K", "IQ4_NL", "IQ4_XS", "MXFP4"}
    read = write = deq = 0
    for t in doc["tensors"]:
        elems = 1
        for d in t["shape_ggml_order"]:
            elems *= d
        read += t["n_bytes"]
        write += elems * width
        if t["ggml_type"] in quant:
            deq += 1
    return {"read_bytes": read, "write_bytes": write, "dequantised": deq,
            "expansion": write / read}


def near(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))
