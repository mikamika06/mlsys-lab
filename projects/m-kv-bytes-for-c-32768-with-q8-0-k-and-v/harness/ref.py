import struct


def pack_gguf_metadata(kv_dict: dict, arch: str = "llama") -> bytes:
    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<III", 3, 0, len(kv_dict)))

    def _pack_str(s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    type_uint32 = 4

    for k, v in kv_dict.items():
        full_key = f"{arch}.{k}"
        buf.extend(_pack_str(full_key))
        buf.extend(struct.pack("<I", type_uint32))
        buf.extend(struct.pack("<I", int(v)))

    return bytes(buf)


GGUF_TEST_CASES = [
    {
        "block_count": 32,
        "feed_forward_length": 14336,
        "embedding_length": 4096,
        "head_count": 32,
        "head_count_kv": 8,
        "context_length": 32768,
    },
    {
        "block_count": 80,
        "feed_forward_length": 28672,
        "embedding_length": 8192,
        "head_count": 64,
        "head_count_kv": 8,
        "context_length": 32768,
    },
    {
        "block_count": 24,
        "feed_forward_length": 8192,
        "embedding_length": 2048,
        "head_count": 16,
        "head_count_kv": 16,
        "context_length": 16384,
    },
]

BINARY_FIXTURES = [pack_gguf_metadata(tc) for tc in GGUF_TEST_CASES]


def reference_calculate_kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, quant_type="f16"):
    num_elements = 2 * n_layers * n_kv_heads * head_dim * seq_len
    q = quant_type.lower()
    if q == "f16":
        return num_elements * 2
    elif q == "q8_0":
        return ((num_elements + 31) // 32) * 34
    elif q == "q4_0":
        return ((num_elements + 31) // 32) * 18
    raise ValueError("Invalid quant")


def reference_evaluate_perplexity_delta(base_ppl, quant_type, seq_len):
    q = quant_type.lower()
    if q == "f16":
        return 0.0
    factor = 1.0 + (seq_len / 32768.0) * 0.5
    if q == "q8_0":
        return round(0.015 * factor, 4)
    elif q == "q4_0":
        return round(0.180 * factor, 4)
    raise ValueError("Invalid quant")
