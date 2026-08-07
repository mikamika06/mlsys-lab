import sys
import numpy as np


def check(workdir):
    sys.path.insert(0, workdir)
    res = {"export_ok": 0.0, "dynamic_shapes_valid": 0.0}

    try:
        from exporter.export_pipeline import export_model_with_dynamic_shapes
        import ref

        model = ref.ReferenceModel()
        sample_input = {"x": np.zeros((2, 16, 32)), "weight": np.zeros((32, 64))}
        dynamic_shapes = {"batch": (1, 32), "seq_len": (1, 128)}

        prog = export_model_with_dynamic_shapes(model, sample_input, dynamic_shapes)
        if isinstance(prog, dict) and prog.get("status") == "exported":
            res["export_ok"] = 1.0

        try:
            bad_input = {"x": np.zeros((64, 16, 32)), "weight": np.zeros((32, 64))}
            export_model_with_dynamic_shapes(model, bad_input, dynamic_shapes)
        except ValueError:
            res["dynamic_shapes_valid"] = 1.0

    except Exception:
        pass

    return res
