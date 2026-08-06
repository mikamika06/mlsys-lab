import importlib.util
import os


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    if not fns:
        return None
    for fn in fns:
        fn()
    return True


def _survives(path):
    try:
        return _run(path) is True
    except Exception:
        return False


def check(workdir):
    out = {"has_tests": 0.0, "passes_on_good": 0.0, "catches_bad_is_cl": 0.0, "catches_bad_pipeline": 0.0}
    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = f"tests fail on a correct implementation: {type(e).__name__}: {str(e)[:120]}"
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import layout.pipeline as pl
    import layout.strides as st

    good_is_cl = st.is_channels_last

    def bad_is_cl(shape, strides):
        return strides[1] == 1

    st.is_channels_last = bad_is_cl
    try:
        out["catches_bad_is_cl"] = 0.0 if _survives(path) else 1.0
    finally:
        st.is_channels_last = good_is_cl

    good_pipe = pl.steady_state_batch_time

    def bad_pipe(cpu, xfer, gpu, pin, nb):
        if nb:
            return max(cpu, xfer + gpu)
        return max(cpu + xfer, gpu)

    pl.steady_state_batch_time = bad_pipe
    try:
        out["catches_bad_pipeline"] = 0.0 if _survives(path) else 1.0
    finally:
        pl.steady_state_batch_time = good_pipe

    return out
