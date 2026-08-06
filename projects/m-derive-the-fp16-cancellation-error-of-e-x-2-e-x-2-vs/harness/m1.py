import ref

def check(workdir):
    from variance.derivation import compute_fp16_variance
    data, _ = ref.generate_fixtures()
    got = compute_fp16_variance(data)
    want = ref.compute_fp16_variance_reference(data)
    rel_err = float(abs(got - want) / (abs(want) + 1e-6))
    return {"rel_err": rel_err}
