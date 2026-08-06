import re

BLOCK = re.compile(r"^blk\.(\d+)\.(.+)$")

PER_LAYER = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
}

EXPERTS = {
    "ffn_gate_exps.weight": "mlp.experts.{e}.gate_proj.weight",
    "ffn_up_exps.weight": "mlp.experts.{e}.up_proj.weight",
    "ffn_down_exps.weight": "mlp.experts.{e}.down_proj.weight",
}

TOP = {
    "token_embd.weight": "embed_tokens.weight",
    "output_norm.weight": "norm.weight",
    "output.weight": "lm_head.weight",
}


def to_target(name, experts=0):
    """The target name for one GGUF tensor.

    A string for the usual one-to-one case, a list when a fused expert tensor
    fans out, None when the tensor has no counterpart on the other side.
    """
    if name in TOP:
        return TOP[name]
    m = BLOCK.match(name)
    if not m:
        return None
    layer, tail = int(m.group(1)), m.group(2)
    if tail in PER_LAYER:
        return "layers.%d.%s" % (layer, PER_LAYER[tail])
    if tail in EXPERTS:
        if not experts:
            return None
        return ["layers.%d.%s" % (layer, EXPERTS[tail].format(e=e))
                for e in range(experts)]
    return None


def map_index(tensors, experts=0):
    mapped, fanned, unmapped = {}, {}, []
    for t in tensors:
        target = to_target(t["name"], experts)
        if target is None:
            unmapped.append(t["name"])
        elif isinstance(target, list):
            fanned[t["name"]] = target
        else:
            mapped[t["name"]] = target
    return {"mapped": mapped, "fanned_out": fanned, "unmapped": unmapped,
            "target_count": len(mapped) + sum(len(v) for v in fanned.values())}


def layer_targets(mapping, layer):
    want = "layers.%d." % layer
    out = set()
    for target in mapping["mapped"].values():
        if target.startswith(want):
            out.add(target)
    for targets in mapping["fanned_out"].values():
        for target in targets:
            if target.startswith(want):
                out.add(target)
    return out
