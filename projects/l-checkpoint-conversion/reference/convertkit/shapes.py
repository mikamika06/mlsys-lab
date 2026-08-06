QUANTISED = {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K",
             "Q5_K", "Q6_K", "Q8_K", "IQ4_NL", "IQ4_XS", "MXFP4"}


def target_shape(ggml_shape):
    """GGUF stores the fastest-moving dimension first; the target stores rows
    first, so the shape is the reverse."""
    return list(reversed(list(ggml_shape)))


def is_quantised(ggml_type):
    return ggml_type in QUANTISED


def attention_shapes(meta, prefix):
    hidden = meta["%s.embedding_length" % prefix]
    heads = meta["%s.attention.head_count" % prefix]
    kv_heads = meta["%s.attention.head_count_kv" % prefix]
    # hidden // heads is a guess, not a rule. This checkpoint carries 5120
    # hidden across 32 heads and still uses a head dimension of 128, so a
    # converter that divides produces q_proj rows that do not exist.
    head_dim = meta.get("%s.attention.key_length" % prefix) or hidden // heads
    return {"hidden": hidden, "heads": heads, "kv_heads": kv_heads,
            "head_dim": head_dim,
            "q_out": heads * head_dim, "kv_out": kv_heads * head_dim,
            "grouped": kv_heads != heads,
            "group_size": heads // kv_heads if kv_heads else 0}


def check_attention(tensors, meta, prefix, layer=0):
    want = attention_shapes(meta, prefix)
    problems = []
    by_name = {t["name"]: t for t in tensors}
    for tail, rows in (("attn_q", want["q_out"]), ("attn_k", want["kv_out"]),
                       ("attn_v", want["kv_out"])):
        name = "blk.%d.%s.weight" % (layer, tail)
        t = by_name.get(name)
        if not t:
            problems.append("%s is missing" % name)
            continue
        shape = target_shape(t["shape_ggml_order"])
        if shape != [rows, want["hidden"]]:
            problems.append("%s: shape %s, the metadata implies %s"
                            % (name, shape, [rows, want["hidden"]]))
    return {"expected": want, "problems": problems}
