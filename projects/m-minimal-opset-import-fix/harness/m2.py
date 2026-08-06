import os
import onnx
import ref


def check(workdir):
    from opsetfix.transformer import fix_opset_and_clip

    os.makedirs(os.path.join(workdir, "build"), exist_ok=True)
    in_path = os.path.join(workdir, "build", "input_m2.onnx")
    out_path = os.path.join(workdir, "build", "output_m2.onnx")
    ref.create_sample_model(in_path)

    try:
        fix_opset_and_clip(in_path, out_path, target_opset=13)
    except Exception as e:
        return {"clips_rewritten": 0.0, "_note": f"Execution failed: {type(e).__name__}: {str(e)}"}

    if not os.path.isfile(out_path):
        return {"clips_rewritten": 0.0, "_note": "Output model file was not created"}

    model = onnx.load(out_path)
    rewritten = 0
    for node in model.graph.node:
        if node.op_type == "Clip":
            has_attr = any(attr.name in ("min", "max") for attr in node.attribute)
            has_inputs = len(node.input) >= 2
            if not has_attr and has_inputs:
                rewritten = 1.0

    return {"clips_rewritten": rewritten}
