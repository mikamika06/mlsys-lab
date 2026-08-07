import ref


def check(workdir):
    from quantutil.core import select_config
    got = select_config(0.0001)
    want = ref.select_config(0.0001)
    ok = (got == want)
    return {"config_matched": 1.0 if ok else 0.0}
