def encode_with_fallback(text: str, vocab: dict[str, int]) -> list[int]:
    tokens = []
    i = 0
    n = len(text)
    sorted_vocab = sorted(vocab.keys(), key=len, reverse=True)

    while i < n:
        matched = False
        for tok in sorted_vocab:
            if tok.startswith("<0x") and tok.endswith(">"):
                continue
            if text.startswith(tok, i):
                tokens.append(vocab[tok])
                i += len(tok)
                matched = True
                break
        if not matched:
            char_bytes = text[i].encode("utf-8")
            for b in char_bytes:
                byte_tok = f"<0x{b:02X}>"
                if byte_tok in vocab:
                    tokens.append(vocab[byte_tok])
                else:
                    raise ValueError(f"Byte token {byte_tok} missing from vocabulary")
            i += 1
    return tokens


def decode_with_fallback(token_ids: list[int], inv_vocab: dict[int, str]) -> str:
    res = []
    byte_buf = bytearray()

    def flush_bytes():
        nonlocal byte_buf
        if byte_buf:
            res.append(byte_buf.decode("utf-8", errors="replace"))
            byte_buf = bytearray()

    for tid in token_ids:
        tok = inv_vocab.get(tid, "")
        if tok.startswith("<0x") and tok.endswith(">") and len(tok) == 6:
            try:
                b_val = int(tok[3:5], 16)
                byte_buf.append(b_val)
                continue
            except ValueError:
                pass

        flush_bytes()
        res.append(tok)

    flush_bytes()
    return "".join(res)


def verify_round_trip(text: str, vocab: dict[str, int]) -> bool:
    inv_vocab = {v: k for k, v in vocab.items()}
    encoded = encode_with_fallback(text, vocab)
    decoded = decode_with_fallback(encoded, inv_vocab)
    return decoded == text
