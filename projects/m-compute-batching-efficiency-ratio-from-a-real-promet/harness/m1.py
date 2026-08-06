import ref


def check(workdir):
    from batching.efficiency import compute_batching_efficiency
    out = {"rel_err": 1.0}
    dump = ref.generate_fixture(seed=101)
    want = ref.compute_batching_efficiency(dump)
    try:
        got = compute_batching_efficiency(dump)
    except Exception as e:
        out["_note"] = f"raised exception: {e}"
        return out

    if want == 0.0:
        err = 0.0 if got == 0.0 else 1.0
    else:
        err = abs(got - want) / abs(want)
    out["rel_err"] = float(err)
    return out
