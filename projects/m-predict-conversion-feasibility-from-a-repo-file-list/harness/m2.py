import ref


def check(workdir):
    from ggufconv.memory import estimate_conversion_memory
    from ggufconv.tokenizer import compute_chkhsh

    out = {
        "chkhsh_matched": 0.0,
        "lazy_mem_matched": 0.0,
        "eager_mem_matched": 0.0,
        "lazy_saves_memory": 0.0,
    }

    tok_ok = True
    for tokens, pre_tok, want_hash in ref.TOKENIZER_TEST_CASES:
        got_hash = compute_chkhsh(tokens, pre_tok)
        if got_hash != want_hash:
            tok_ok = False
            out["_note"] = f"chkhsh mismatch: got {got_hash}, want {want_hash}"
            break
    if tok_ok:
        out["chkhsh_matched"] = 1.0

    for tensors, want_lazy, want_eager in ref.MEMORY_TEST_CASES:
        got_lazy = estimate_conversion_memory(tensors, lazy=True)
        got_eager = estimate_conversion_memory(tensors, lazy=False)

        if got_lazy == want_lazy:
            out["lazy_mem_matched"] = 1.0
        elif "_note" not in out:
            out["_note"] = f"lazy memory mismatch: got {got_lazy}, want {want_lazy}"

        if got_eager == want_eager:
            out["eager_mem_matched"] = 1.0
        elif "_note" not in out:
            out["_note"] = f"eager memory mismatch: got {got_eager}, want {want_eager}"

        if (
            got_lazy.get("peak_memory_bytes", 0)
            < got_eager.get("peak_memory_bytes", 0)
        ):
            out["lazy_saves_memory"] = 1.0

    return out
