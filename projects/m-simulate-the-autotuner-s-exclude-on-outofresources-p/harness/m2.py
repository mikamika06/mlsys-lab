import sys
import ref


def check(workdir):
    sys.path.insert(0, workdir)
    out = {"argmin_matched": 0.0, "evals_matched": 0.0}
    try:
        import autotune
        configs = ref.generate_configs()
        rk = ["BLOCK_M", "BLOCK_N", "num_warps"]

        for limit in [16384, 32768, 999999]:
            eval_ref, met_ref = ref.make_evaluator(limit)
            want_idx = ref.autotune(configs, eval_ref, rk)

            eval_got, met_got = ref.make_evaluator(limit)
            try:
                # Patch module's OutOfResources to match user's exceptions in evaluate
                autotune.OutOfResources = ref.OutOfResources
                got_idx = autotune.autotune(configs, eval_got, rk)

                if got_idx == want_idx:
                    out["argmin_matched"] += 1.0
                if met_got["evals"] == met_ref["evals"]:
                    out["evals_matched"] += 1.0
            except Exception:
                pass
    except Exception:
        pass
    finally:
        sys.path.pop(0)

    return out
