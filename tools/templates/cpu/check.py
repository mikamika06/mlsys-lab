from mlsys.sim import cache as cachesim

def grade(sol, fx) -> dict:
    n, EB = 64, 8  # 64x64 float64 matrix, row-major
    try:
        order = list(sol.traverse(n))
    except Exception:
        return {"covers_all": 0.0, "misses": 10**9}
    covers = 1.0 if sorted(order) == list(range(n * n)) else 0.0
    addrs = [i * EB for i in order]
    misses = cachesim.simulate(addrs, line_bytes=64, sets=64, ways=8)["misses"]
    return {"covers_all": covers, "misses": misses}
