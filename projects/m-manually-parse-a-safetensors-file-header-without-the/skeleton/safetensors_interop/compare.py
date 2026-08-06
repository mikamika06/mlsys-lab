"""Format comparison utilities for safetensors and GGUF."""


def parse_gguf_bytes(gguf_bytes: bytes, alignment: int = 32) -> dict:
    raise NotImplementedError


def verify_f16_bit_identity(st_bytes: bytes, gguf_bytes: bytes) -> dict:
    raise NotImplementedError
