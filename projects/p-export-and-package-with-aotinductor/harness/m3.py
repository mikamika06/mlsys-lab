import sys
import os


def check(workdir):
    sys.path.insert(0, workdir)
    res = {"artifact_built": 0.0, "so_file_exists": 0.0}

    try:
        from exporter.aot_builder import compile_aot_artifact

        exported_program = {"status": "exported"}
        target_path = os.path.join(workdir, "build", "model_aot.so")

        out_path = compile_aot_artifact(exported_program, target_path)
        if out_path == target_path:
            res["artifact_built"] = 1.0

        if os.path.isfile(target_path) and os.path.getsize(target_path) > 0:
            res["so_file_exists"] = 1.0

    except Exception:
        pass

    return res
