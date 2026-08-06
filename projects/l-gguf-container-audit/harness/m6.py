import ref


def check(workdir):
    import importlib

    mod = importlib.import_module("tests.test_container")
    out = {"accepts_clean": 0.0, "rejects_corrupt": 0.0, "message_names_tensor": 0.0}
    if not hasattr(mod, "assert_loadable"):
        return out
    try:
        mod.assert_loadable(ref.clean_blob())
        out["accepts_clean"] = 1.0
    except AssertionError:
        return out
    try:
        mod.assert_loadable(ref.corrupt_blob())
    except AssertionError as e:
        out["rejects_corrupt"] = 1.0
        if ref.corruption()["damaged_tensor"] in str(e):
            out["message_names_tensor"] = 1.0
    return out
