def check(workdir):
    import ref
    m = {"index_ok": 0.0}
    prompt = [1, 2, 3, 4, 5, 1, 2, 3, 4, 6]
    idx = ref.create_oracle_index(prompt, n=3)
    res = idx.lookup([1, 2, 3])
    if len(res) >= 1 and 4 in res[0]:
        m["index_ok"] = 1.0
    return m
