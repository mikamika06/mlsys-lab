import struct


def make_mock_metadata_samples():
    return [
        {
            "general.architecture": "llama",
            "llama.tokenizer.eos_token_id": 2,
            "tokenizer.ggml.tokens": ["<unk>", "<s>", "</s>"],
            "tokenizer.ggml.token_type": [2, 1, 3]
        },
        {
            "general.architecture": "mistral",
            "mistral.tokenizer.eos_token_id": 2,
            "tokenizer.ggml.tokens": ["<unk>", "<s>", "[PAD]", "</s>"],
            "tokenizer.ggml.token_type": [2, 1, 1, 3]
        },
        {
            "general.architecture": "qwen2",
            "qwen2.tokenizer.eos_token_id": 151643,
            "tokenizer.ggml.tokens": [f"t{i}" for i in range(10)],
            "tokenizer.ggml.token_type": [1] * 10
        }
    ]


def make_binary_gguf_header(ctx_len=4096):
    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<I", 3))
    buf.extend(struct.pack("<Q", 0))
    buf.extend(struct.pack("<Q", 1))

    key = b"llm.context_length"
    buf.extend(struct.pack("<Q", len(key)))
    buf.extend(key)
    buf.extend(struct.pack("<I", 4))
    buf.extend(struct.pack("<I", ctx_len))
    return buf


def make_rope_samples():
    return [
        {
            "general.architecture": "llama",
            "llama.rope.freq_base": 500000.0,
            "llama.rope.scaling.type": "linear",
            "llama.rope.scaling.factor": 2.0,
            "llama.rope.scaling.original_context_length": 4096
        },
        {
            "general.architecture": "qwen2",
            "rope.freq_base": 1000000.0,
            "rope.scaling.type": "none",
            "rope.scaling.factor": 1.0,
            "rope.scaling.original_context_length": 0
        }
    ]
