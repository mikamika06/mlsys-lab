import os
import onnx
import ref


def check(workdir):
    from opsetfix.transformer import fix_opset_and_clip

    os.makedirs(os.path.join(workdir, "build"), exist_ok=True)
    in_path = os.path.join(workdir, "build", "input_m1.onnx")
    out_path = os.path.join(workdir, "build", "output_m1.onnx")
    ref.create_sample_model(in_path)

    try:
        fix_opset_and_clip(in_path, out_path, target_opset=13)
    except Exception as e:
        return {"opserset_aligned": 0.0, "_note": f"Execution failed: {type(e).__name__}: {str(e)}"}

    if not os.path.isfile(out_path):
        return {"opserset_aligned": 0.0, "_note": "Output model file was not created"}

    model = onnx.load(out_path)
    aligned = 0
    for opset in model.opset_import:
        if (opset.domain == "" or opset.domain == "ai.onnx") and opset.version >= 13:
            aligned = 1.0

    return {"opserset_aligned": aligned}
