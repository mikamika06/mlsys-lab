def check(workdir):
    import ref
    m = {"candidates_ok": 0.0}
    prompt = [10, 20, 30, 40, 50, 10, 20, 30, 70]
    idx = ref.create_oracle_index(prompt, n=3)
    cands = ref.oracle_select(idx, [10, 20, 30], k=2)
    if len(cands) == 2:
        m["candidates_ok"] = 1.0
    return m
