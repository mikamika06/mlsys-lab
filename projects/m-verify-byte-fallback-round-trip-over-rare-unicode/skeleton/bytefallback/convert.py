def encode_with_fallback(text: str, vocab: dict[str, int]) -> list[int]:
    """Encode text using vocab tokens or byte fallback <0xXX>."""
    raise NotImplementedError


def decode_with_fallback(token_ids: list[int], inv_vocab: dict[int, str]) -> str:
    """Decode token IDs back to text, grouping adjacent byte tokens into raw UTF-8 bytes."""
    raise NotImplementedError


def verify_round_trip(text: str, vocab: dict[str, int]) -> bool:
    """Verify that text -> encode -> decode recovers the exact input string."""
    raise NotImplementedError
