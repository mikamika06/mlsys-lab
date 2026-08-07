import sys

sys.path.insert(0, ".")
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    from memrunner.bandwidth import predict_decode_tok_s
    from memrunner.kquants import explain_kquant_precision_mix

    kquant_ok = 1
    for cfg in ref.CONFIGS:
        exp_want = ref.explain_kquant_precision_mix(cfg)
        exp_got = explain_kquant_precision_mix(cfg)
        if (
            exp_got.get("quant_type") != exp_want["quant_type"]
            or exp_got.get("attn_bits") != exp_want["attn_bits"]
            or exp_got.get("ffn_down_bits") != exp_want["ffn_down_bits"]
            or exp_got.get("is_mixed_precision") != exp_want["is_mixed_precision"]
        ):
            kquant_ok = 0
            break

    max_toks_err = 0.0
    bandwidths = [800, 1000, 1600]
    for i, cfg in enumerate(ref.CONFIGS):
        bw = bandwidths[i]
        seq = 2048
        bs = 1
        want_toks = ref.predict_decode_tok_s(cfg, seq, bw, bs)
        got_toks = predict_decode_tok_s(cfg, seq, bw, bs)
        err = abs(got_toks - want_toks) / float(want_toks)
        if err > max_toks_err:
            max_toks_err = err

    return {"kquant_matched": float(kquant_ok), "toks_rel_err": float(max_toks_err)}
