import ref


def check(workdir):
    from ppl.chunking import compute_perplexity

    out = {"ppl_rel_err": 1.0, "chunks_evaluated": 0.0}
    tokens = ref.generate_tokens(num_tokens=150, vocab_size=32, seed=7)
    model = ref.SyntheticModel(vocab_size=32, seed=11)
    chunk_size = 32

    tracked_chunks = []

    def tracking_model(chunk):
        tracked_chunks.append(len(chunk))
        return model(chunk)

    want = ref.ref_compute_perplexity(model, tokens, chunk_size)
    try:
        got = compute_perplexity(tracking_model, tokens, chunk_size)
    except Exception as e:
        out["_note"] = f"compute_perplexity raised: {type(e).__name__}: {e}"
        return out

    out["chunks_evaluated"] = float(len(tracked_chunks))

    if want > 0:
        rel_err = abs(got - want) / want
        out["ppl_rel_err"] = float(rel_err)
        if rel_err > 1e-4:
            out["_note"] = f"got PPL {got}, expected {want} (rel_err={rel_err})"
    else:
        out["ppl_rel_err"] = 0.0 if got == want else 1.0

    return out
