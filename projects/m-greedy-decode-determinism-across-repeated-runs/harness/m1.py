import ref


def check(workdir):
    from decoder.pipeline import run_greedy_decode
    from decoder.metrics import compute_match_fraction

    model, tokenizer, prompt = ref.get_mock_setup()
    sequences = run_greedy_decode(model, tokenizer, prompt, runs=3)
    fraction = compute_match_fraction(sequences)

    out = {"exact_match_fraction": float(fraction)}
    if fraction < 1.0:
        out["_note"] = f"Expected exact match fraction 1.0, got {fraction}"
    return out
