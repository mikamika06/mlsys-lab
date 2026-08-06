import ref


def check(workdir):
    from streamkv.metrics import evaluate_perplexity_curves

    seq_lens, full_ppl, sink_ppl, random_ppl = ref.get_perplexity_data()
    want = ref.evaluate_perplexity_curves(seq_lens, full_ppl, sink_ppl, random_ppl)
    got = evaluate_perplexity_curves(seq_lens, full_ppl, sink_ppl, random_ppl)

    match = 1.0 if got == want else 0.0
    return {"perplexity_trend_matched": match}
