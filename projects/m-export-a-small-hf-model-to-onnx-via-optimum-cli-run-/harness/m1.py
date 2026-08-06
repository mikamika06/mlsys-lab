import ref
import numpy as np


def check(workdir):
    from edgeonnx.export import parse_export_config, simulate_export
    from edgeonnx.runner import run_inference

    max_err = 0.0
    for cfg in ref.CONFIGS:
        parsed = parse_export_config(cfg)
        spec = simulate_export(parsed)
        inputs = np.array([0.5, 1.0, 1.5], dtype=np.float32)

        got = run_inference(spec, inputs, "CoreMLExecutionProvider")
        want = ref.compute_output(spec, inputs)

        err = float(np.max(np.abs(got - want)))
        if err > max_err:
            max_err = err

    return {"max_abs_err": float(max_err)}
