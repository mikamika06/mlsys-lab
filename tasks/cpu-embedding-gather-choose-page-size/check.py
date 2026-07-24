from mlsys.sim import cache as cachesim

def _ref(indices, row_bytes, page_sizes):
    addrs = [i * row_bytes for i in indices]
    best = None
    best_misses = None
    for p in page_sizes:
        res = cachesim.simulate(addrs, line_bytes=p, sets=64, ways=1)
        misses = res["misses"]
        if best is None or misses < best_misses or (misses == best_misses and p < best):
            best = p
            best_misses = misses
    return best

def grade(sol, fx) -> dict:
    tests = [
        ([1,2,4,8,16,32,64,128,256], 64, [4096, 2**21, 2**12]),
        ([0,1,2,3,4,0,1,2,3,4], 64, [4096, 2**21]),
        ([1,2,8,1,4096,8192,4096], 64, [4096, 2**21]),
        ([i*37 % 10000 for i in range(200)], 128, [4096, 2**21, 2**12]),
        ([0,1024,2048,3072,4096,5120,6144,7168], 256, [4096, 8192, 2**21]),
    ]
    ok = 1.0
    for indices, row_bytes, page_sizes in tests:
        try:
            ans_ref = _ref(indices, row_bytes, page_sizes)
            ans_student = sol.choose_page_size(list(indices), row_bytes, list(page_sizes))
        except Exception:
            ok = 0.0
            break
        if ans_ref != ans_student:
            ok = 0.0
            break
    return {"exact_match": ok}
