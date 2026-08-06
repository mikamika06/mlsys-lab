import ref


def check(workdir):
    from packeff.crossover import compute_attention_costs
    cases = ref.get_test_cases()
    ok = 0
    total = len(cases)
    for lengths, max_len in cases:
        want = ref.compute_attention_costs(lengths, max_len)
        try:
            got = compute_attention_costs(lengths, max_len)
            if (isinstance(got, dict) and
                abs(got.get("padding_cost", 0) - want["padding_cost"]) < 1e-5 and
                abs(got.get("packing_cost", 0) - want["packing_cost"]) < 1e-5):
                ok += 1
        except Exception:
            pass
    return {"crossover_matched": 1.0 if ok == total else 0.0}
