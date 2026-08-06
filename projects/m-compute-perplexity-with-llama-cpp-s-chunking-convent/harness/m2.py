import ref


def check(workdir):
    from ppl.metrics import compute_logit_metrics

    out = {"kld_rel_err": 1.0, "top1_match": 0.0}

    model_base = ref.SyntheticModel(vocab_size=64, seed=10, noise_std=0.0)
    model_quant = ref.SyntheticModel(vocab_size=64, seed=10, noise_std=0.5)

    tokens = ref.generate_tokens(num_tokens=100, vocab_size=64, seed=20)
    chunk = tokens[:64]

    base_logits = model_base(chunk)
    quant_logits = model_quant(chunk)

    want = ref.ref_compute_logit_metrics(base_logits, quant_logits)

    try:
        got = compute_logit_metrics(base_logits, quant_logits)
    except Exception as e:
        out["_note"] = f"compute_logit_metrics raised: {type(e).__name__}: {e}"
        return out

    if not isinstance(got, dict) or "mean_kld" not in got or "top1_agreement" not in got:
        out["_note"] = f"expected dict with 'mean_kld' and 'top1_agreement', got {got}"
        return out

    want_kld = want["mean_kld"]
    got_kld = got["mean_kld"]
    rel_err = abs(got_kld - want_kld) / (want_kld + 1e-9)
    out["kld_rel_err"] = float(rel_err)

    want_top1 = want["top1_agreement"]
    got_top1 = got["top1_agreement"]
    if abs(got_top1 - want_top1) < 1e-6:
        out["top1_match"] = 1.0
    else:
        out["_note"] = f"top1 agreement mismatch: got {got_top1}, expected {want_top1}"

    return out
