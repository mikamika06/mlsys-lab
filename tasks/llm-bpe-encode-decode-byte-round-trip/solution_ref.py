def bpe_encode(text: str) -> list[tuple[int, int]]:
    """
    Encode a string into run‑length pairs of UTF‑8 bytes.
    Each pair is (byte_value, count), where count >= 1.
    """
    data = text.encode("utf-8")
    tokens: list[tuple[int, int]] = []
    i = 0
    while i < len(data):
        val = data[i]
        cnt = 1
        j = i + 1
        while j < len(data) and data[j] == val:
            cnt += 1
            j += 1
        tokens.append((int(val), int(cnt)))
        i = j
    return tokens


def bpe_decode(tokens: list[tuple[int, int]]) -> str:
    """
    Decode a token list produced by `bpe_encode` back to the original string.
    """
    byte_list = []
    for val, cnt in tokens:
        byte_list.extend([val] * cnt)
    return bytes(byte_list).decode("utf-8")
