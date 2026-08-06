import ref


def check(workdir):
    import sys
    sys.path.insert(0, workdir)
    from llambench.parser import extract_quant_metrics, parse_llama_bench_json

    out = {"parsed_configs_matched": 0.0}
    matched = 0

    for item in ref.TEST_BENCHMARKS:
        raw = item["raw_json"]
        got_parsed = parse_llama_bench_json(raw)
        for qtype in ["Q4_K_M", "Q8_0"]:
            want_m = ref.reference_extract(ref.reference_parse(raw), qtype)
            got_m = extract_quant_metrics(got_parsed, qtype)
            if abs(got_m["pp"] - want_m["pp"]) < 1e-3 and abs(got_m["tg"] - want_m["tg"]) < 1e-3:
                matched += 1
            elif "_note" not in out:
                out["_note"] = f"Quant {qtype} mismatch: got {got_m}, expected {want_m}"

    out["parsed_configs_matched"] = float(matched)
    return out
