import sys


def check(workdir):
    sys.path.insert(0, workdir)
    res = {"custom_op_registered": 0.0, "abstract_impl_ok": 0.0}

    try:
        from exporter.custom_ops import CustomOpRegistry

        registry = CustomOpRegistry()
        registry.register()
        if registry.registered:
            res["custom_op_registered"] = 1.0

        out_shape = registry.meta_impl((2, 10, 32), (32, 64))
        if out_shape == (2, 10, 64):
            res["abstract_impl_ok"] = 1.0

    except Exception:
        pass

    return res
