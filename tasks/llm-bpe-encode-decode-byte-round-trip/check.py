import numpy as np
from mlsys.scorers import byte_exact_fraction

def _oracle_encode(text: str):
    """Reference run‑length encoder used by the grader."""
    b = text.encode('utf-8')
    tokens = []
    i = 0
    while i < len(b):
        val = b[i]
        count = 1
        j = i + 1
        while j < len(b) and b[j] == val:
            count += 1
            j += 1
        tokens.append((int(val), int(count)))
        i = j
    return tokens

def grade(sol, fx) -> dict:
    # Test cases – a mix of ASCII, Unicode, empty string, long runs.
    tests = [
        "",
        "hello world",
        "aaaaa",
        "abcabcabc",
        "😀😃😄😁",
        "The quick brown fox jumps over the lazy dog.",
        "🚀🌕✨🔥💧🌀",
        "a" * 50 + "b" * 30 + "c" * 20,
    ]

    byte_ok = 1.0
    token_ok = 1.0

    for text in tests:
        try:
            # Candidate round‑trip
            cand_tokens = sol.bpe_encode(text)
            cand_decoded = sol.bpe_decode(cand_tokens)

            orig_bytes = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
            cand_bytes = np.frombuffer(cand_decoded.encode('utf-8'), dtype=np.uint8)

            # Byte‑exactness – handle empty case specially
            if orig_bytes.size == 0 and cand_bytes.size == 0:
                pass  # ok
            else:
                if byte_exact_fraction(orig_bytes, cand_bytes) < 1.0:
                    byte_ok = 0.0

            # Token exactness against oracle
            ref_tokens = _oracle_encode(text)
            if cand_tokens != ref_tokens:
                token_ok = 0.0

        except Exception:
            byte_ok = 0.0
            token_ok = 0.0
            break

    return {
        "byte_exact_fraction": byte_ok,
        "token_exact_fraction": token_ok,
    }
