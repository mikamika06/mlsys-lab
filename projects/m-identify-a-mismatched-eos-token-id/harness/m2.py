import ref


def check(workdir):
    from gguf_meta.context import update_context_length_inplace
    from gguf_meta.rope import decode_rope_config

    out = {"context_updates_correct": 0.0, "rope_decodes_correct": 0.0}

    header = ref.make_binary_gguf_header(4096)
    updated = update_context_length_inplace(header, 16384)

    import struct
    new_val = struct.unpack_from("<I", updated, 24 + 8 + len("llm.context_length") + 4)[0]
    if new_val == 16384 and len(updated) == len(header):
        out["context_updates_correct"] = 1.0
    else:
        out["_note"] = f"Context length update failed: got {new_val}"

    rope_samples = ref.make_rope_samples()
    decoded = [decode_rope_config(s) for s in rope_samples]

    if (decoded[0]["freq_base"] == 500000.0 and decoded[0]["scaling_type"] == "linear" and
        decoded[1]["freq_base"] == 1000000.0 and decoded[1]["scaling_type"] == "none"):
        out["rope_decodes_correct"] = 1.0
    else:
        out["_note"] = f"RoPE decode mismatch: {decoded}"

    return out
