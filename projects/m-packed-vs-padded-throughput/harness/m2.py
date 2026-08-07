import ref

def check(workdir):
    try:
        from seqpack.core import padded_cost
        from seqpack.simulate import throughput_ratio, misspecification_effects
    except ImportError:
        return {"padded_cost_matched": 0.0, "throughput_ratio_matched": 0.0, "effects_matched": 0.0}

    out = {
        "padded_cost_matched": 0.0,
        "throughput_ratio_matched": 0.0,
        "effects_matched": 0.0
    }
    pad_ok = 0
    tr_ok = 0
    eff_ok = 0

    for seqlens, block_size, prov_max in ref.FIXTURES:
        bs = len(seqlens)
        seq_len = max(seqlens) if seqlens else 0

        try:
            if padded_cost(bs, seq_len, block_size) == ref.padded_cost(bs, seq_len, block_size):
                pad_ok += 1
        except Exception:
            pass

        try:
            got_tr = throughput_ratio(seqlens, block_size)
            want_tr = ref.throughput_ratio(seqlens, block_size)
            if abs(got_tr - want_tr) < 1e-6:
                tr_ok += 1
        except Exception:
            pass

        try:
            if prov_max >= seq_len:
                got_eff = misspecification_effects(seqlens, block_size, prov_max)
                want_eff = ref.misspecification_effects(seqlens, block_size, prov_max)
                if got_eff["wasted_flops"] == want_eff["wasted_flops"] and \
                   abs(got_eff["relative_degradation"] - want_eff["relative_degradation"]) < 1e-6:
                    eff_ok += 1
        except Exception:
            pass

    out["padded_cost_matched"] = float(pad_ok) / len(ref.FIXTURES)
    out["throughput_ratio_matched"] = float(tr_ok) / len(ref.FIXTURES)
    out["effects_matched"] = float(eff_ok) / len(ref.FIXTURES)
    return out
