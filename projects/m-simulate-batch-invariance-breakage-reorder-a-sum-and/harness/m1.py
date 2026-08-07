import ref

def check(workdir):
    from batchsim.simulate import simulate_reordered_sum
    max_err = 0.0
    for mat in ref.SIMULATE_INPUTS:
        want = ref.simulate_reordered_sum(mat)
        got = simulate_reordered_sum(mat)
        err = abs(want["delta"] - got["delta"])
        if err > max_err:
            max_err = err
    return {"max_abs_err": float(max_err)}
