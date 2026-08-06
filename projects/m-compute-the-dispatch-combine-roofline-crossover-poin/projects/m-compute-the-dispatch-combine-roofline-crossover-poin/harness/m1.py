import ref


def check(workdir):
    from moeroof import compute_crossover
    cases = ref.get_test_cases()
    ok = 0
    out = {"crossover_matched": 0.0}
    for i, c in enumerate(cases):
        want = ref.compute_crossover(c["hidden_dim"], c["num_experts"], c["comm_bw"], c["tflops"])
        got = compute_crossover(c["hidden_dim"], c["num_experts"], c["comm_bw"], c["tflops"])
        if got == want:
            ok += 1
        elif "_note" not in out:
            out["_note"] = f"case {i}: got {got}, want {want}"
    if ok == len(cases):
        out["crossover_matched"] = 1.0
    return out
