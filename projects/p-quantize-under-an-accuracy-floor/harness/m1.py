import ref
import sys


def check(workdir):
    m = {"eval_valid": 0.0}
    sys_path_orig = list(sys.path)
    try:
        sys.path.insert(0, workdir)
        import quant.eval as q_eval

        model = ref.ToyModel()
        x, y = ref.generate_dataset()
        res = q_eval.run_eval(model, x, y)
        if isinstance(res, dict) and "accuracy" in res and "mse" in res:
            m["eval_valid"] = 1.0
    except Exception:
        pass
    finally:
        sys.path[:] = sys_path_orig
    return m
