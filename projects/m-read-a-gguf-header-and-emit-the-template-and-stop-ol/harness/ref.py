import struct

SAMPLE_BYTES = b"GGUF" + struct.pack("<I", 3)
BASE_SIZE = 16000000000
BASE_TOK_S = 18.5
BASE_CFG = {"hidden_size": 4096, "num_attention_heads": 32}
ADAPTER_CFG = {"hidden_size": 4096, "num_attention_heads": 32}
