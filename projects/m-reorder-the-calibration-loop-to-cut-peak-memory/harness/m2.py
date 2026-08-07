import ref

def check(workdir):
    from quantcal.limits import max_workable_samples
    got = max_workable_samples(None, None, 300)
    want = ref.get_reference_limit(None, None, 300)
    ok = 1 if got == want else 0
    return {"argmin_index": float(ok)}
