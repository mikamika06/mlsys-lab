SCHEMAS = [
    {
        "required": ["llama.vocab_size", "llama.context_length", "llama.embedding_length"],
        "metadata": {"llama.vocab_size": 32000}
    },
    {
        "required": ["llama.vocab_size", "llama.context_length", "llama.embedding_length"],
        "metadata": {"llama.vocab_size": 32000, "llama.context_length": 4096}
    },
    {
        "required": ["llama.vocab_size", "llama.context_length", "llama.embedding_length"],
        "metadata": {"llama.context_length": 4096, "llama.embedding_length": 4096}
    }
]

METADATAS = [
    {"embedding_length": 4096, "attention.head_count": 32, "attention.head_count_kv": 8},
    {"embedding_length": 2048, "attention.head_count": 16, "attention.head_count_kv": 16},
    {"embedding_length": 8192, "attention.head_count": 64, "attention.head_count_kv": 8}
]
