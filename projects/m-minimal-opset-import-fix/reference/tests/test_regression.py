import sys
import os
import onnx
from onnx import helper
sys.path.insert(0, ".")
from opsetfix.transformer import fix_opset_and_clip


def test_clip_rewrite_structure():
    input_tensor = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3])
    output_tensor = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Clip", ["X"], ["Y"], name="test_clip", min=0.0, max=6.0)
    graph = helper.make_graph([node], "test_graph", [input_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

    os.makedirs("build", exist_ok=True)
    in_path = "build/test_in.onnx"
    out_path = "build/test_out.onnx"
    onnx.save(model, in_path)

    fix_opset_and_clip(in_path, out_path, target_opset=13)

    fixed_model = onnx.load(out_path)
    for opset in fixed_model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            assert opset.version >= 13

    for n in fixed_model.graph.node:
        if n.op_type == "Clip":
            assert len(n.input) >= 2
            for attr in n.attribute:
                assert attr.name not in ("min", "max")


def test_opset_version_is_updated():
    input_tensor = helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2])
    output_tensor = helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2])
    node = helper.make_node("Abs", ["X"], ["Y"])
    graph = helper.make_graph([node], "test_graph_abs", [input_tensor], [output_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 7)])

    os.makedirs("build", exist_ok=True)
    in_path = "build/test_abs_in.onnx"
    out_path = "build/test_abs_out.onnx"
    onnx.save(model, in_path)

    fix_opset_and_clip(in_path, out_path, target_opset=13)
    fixed_model = onnx.load(out_path)

    versions = [op.version for op in fixed_model.opset_import if op.domain == "" or op.domain == "ai.onnx"]
    assert all(v >= 13 for v in versions)
