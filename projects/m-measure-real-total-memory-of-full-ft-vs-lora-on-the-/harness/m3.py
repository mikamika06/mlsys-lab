import importlib.util
import os
import sys


def _run(path):
    spec = importlib.util.spec_from_file_location("learner_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [
        getattr(mod, n)
        for n in dir(mod)
        if n.startswith("test_") and callable(getattr(mod, n))
    ]
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
    out = {
        "has_tests": 0.0,
        "passes_on_good": 0.0,
        "catches_trainable_mismatch": 0.0,
    }
    if workdir not in sys.path:
        sys.path.insert(0, workdir)

    path = os.path.join(workdir, "tests", "test_regression.py")
    if not os.path.isfile(path):
        out["_note"] = "tests/test_regression.py is missing"
        return out

    try:
        first = _run(path)
    except Exception as e:
        out["has_tests"] = 1.0
        out["_note"] = (
            f"the tests fail on correct code: {type(e).__name__}: {str(e)[:120]}"
        )
        return out

    if first is None:
        out["_note"] = "no test_* functions found"
        return out

    out["has_tests"] = 1.0
    out["passes_on_good"] = 1.0

    import ftmem.lora as lora_mod
    import ftmem.memory as mem_mod

    good_lora = lora_mod.count_trainable_params
    good_mem = mem_mod.estimate_memory_footprint

    def buggy_count_trainable_params(config, lora_config=None):
        base_p = lora_mod.count_base_params(config)
        lora_p = (
            lora_mod.count_lora_params(config, lora_config)
            if lora_config
            else base_p
        )
        return lora_p + base_p

    def buggy_estimate_memory_footprint(
        config,
        mode,
        lora_config=None,
        batch_size=1,
        seq_len=512,
        activation_checkpointing=False,
    ):
        res = good_mem(
            config, mode, lora_config, batch_size, seq_len, activation_checkpointing
        )
        if mode == "qlora_4bit":
            res = dict(res)
            res["trainable_params"] = (
                res["trainable_params"] + lora_mod.count_base_params(config)
            )
        return res

    lora_mod.count_trainable_params = buggy_count_trainable_params
    mem_mod.estimate_memory_footprint = buggy_estimate_memory_footprint

    try:
        out["catches_trainable_mismatch"] = 0.0 if _survives(path) else 1.0
    finally:
        lora_mod.count_trainable_params = good_lora
        mem_mod.estimate_memory_footprint = good_mem

    return out
