def check(workdir):
    from bakeoff.runner import BakeoffRunner
    import numpy as np
    m = {"metrics_valid": 0.0, "ratio_bounded": 0.0}
    try:
        runner = BakeoffRunner({"dim": 32})
        inputs = [np.ones((4, 32), dtype=np.float32)]
        res_a = runner.compile_and_run("stack_a", inputs)
        res_b = runner.compile_and_run("stack_b", inputs)
        if "compilation_time" in res_a and "execution_time" in res_a:
            m["metrics_valid"] = 1.0
            ratio = res_a["execution_time"] / (res_b["execution_time"] + 1e-9)
            if 0.01 <= ratio <= 100.0:
                m["ratio_bounded"] = 1.0
    except Exception:
        pass
    return m
