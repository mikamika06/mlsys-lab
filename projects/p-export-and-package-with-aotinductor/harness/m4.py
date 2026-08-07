import sys
import os
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    res = {"num_passed": 0.0, "max_abs_diff": 999.0}

    try:
        from exporter.runtime_runner import StandaloneAOTRunner
        import ref

        so_path = os.path.join(workdir, "build", "model_aot.so")
        os.makedirs(os.path.dirname(so_path), exist_ok=True)
        with open(so_path, "wb") as f:
            f.write(b"\x7fELF_MOCK_DATA")

        runner = StandaloneAOTRunner(so_path)
        ref_model = ref.ReferenceModel()
        inputs_list = ref.generate_test_inputs(seed=42, count=200)

        passed = 0
        max_diff = 0.0

        for inp in inputs_list:
            py_out = ref_model.forward(inp["x"], inp["weight"])
            aot_out = runner.run(inp)

            diff = float(np.max(np.abs(py_out - aot_out)))
            if diff > max_diff:
                max_diff = diff

            if diff <= 1e-4:
                passed += 1

        res["num_passed"] = float(passed)
        res["max_abs_diff"] = float(max_diff)

    except Exception:
        pass

    return res
