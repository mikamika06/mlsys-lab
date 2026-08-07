import ref


def check(workdir):
    from q4k.quant import dequantize_q4_k, quantize_q4_k

    weights_list = ref.get_test_weights()
    matches = 0
    total = len(weights_list)
    for w in weights_list:
        try:
            data1 = quantize_q4_k(w)
            recon = dequantize_q4_k(data1)
            data2 = quantize_q4_k(recon)
            if data1 == data2:
                matches += 1
        except Exception:
            pass

    ratio = float(matches) / float(total) if total > 0 else 0.0
    out = {"exact_match_ratio": ratio}
    if ratio < 1.0:
        out["_note"] = f"Round-trip byte match failed: got {matches}/{total}"
    return out
