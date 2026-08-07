import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"encode_matched": 0.0, "decode_matched": 0.0}

    try:
        from bytefallback.convert import encode_with_fallback, decode_with_fallback
    except ImportError as e:
        out["_note"] = f"Import failure: {e}"
        return out

    vocab = ref.BASE_VOCAB
    inv_vocab = ref.INV_BASE_VOCAB
    test_text = "hello 𓀀 world"

    ref_enc = ref.reference_encode(test_text, vocab)
    try:
        got_enc = encode_with_fallback(test_text, vocab)
        if got_enc == ref_enc:
            out["encode_matched"] = 1.0
        else:
            out["_note"] = f"Encode mismatch. Expected {ref_enc[:5]}..., got {got_enc[:5] if got_enc else got_enc}..."
    except Exception as e:
        out["_note"] = f"Encode raised exception: {e}"
        return out

    ref_dec = ref.reference_decode(ref_enc, inv_vocab)
    try:
        got_dec = decode_with_fallback(ref_enc, inv_vocab)
        if got_dec == ref_dec:
            out["decode_matched"] = 1.0
        else:
            out["_note"] = f"Decode mismatch. Expected {ref_dec!r}, got {got_dec!r}"
    except Exception as e:
        out["_note"] = f"Decode raised exception: {e}"

    return out
