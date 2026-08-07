BASE_VOCAB = {
    "hello": 1,
    "world": 2,
    " ": 3,
    "LLM": 4,
    "test": 5,
    "こんにちは": 6,
}

for i in range(256):
    BASE_VOCAB[f"<0x{i:02X}>"] = 100 + i

INV_BASE_VOCAB = {v: k for k, v in BASE_VOCAB.items()}

RARE_UNICODE_SAMPLES = [
    "hello world",
    "𓀀𓀁𓀂",
    "𠮷野家",
    "👨‍👩‍👧‍👦",
    "hello 𓀀 world",
    "𞸀𞸁𞸂𞸃",
    "A 𐍈 B",
]


def reference_encode(text, vocab):
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
                tokens.append(vocab[byte_tok])
            i += 1
    return tokens


def reference_decode(token_ids, inv_vocab):
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


def reference_verify(text, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    encoded = reference_encode(text, vocab)
    decoded = reference_decode(encoded, inv_vocab)
    return decoded == text
