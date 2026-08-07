import ref
import sys


def check(workdir):
    m = {"mixed_precision_ok": 0.0}
    sys_path_orig = list(sys.path)
    try:
        sys.path.insert(0, workdir)
        import quant.mixed as q_mixed

        model = ref.ToyModel()
        mp = q_mixed.assign_mixed_precision(model)
        if isinstance(mp, list) and len(mp) == len(model.layers):
            m["mixed_precision_ok"] = 1.0
    except Exception:
        pass
    finally:
        sys.path[:] = sys_path_orig
    return m
