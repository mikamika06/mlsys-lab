import os
import onnx
from onnx import helper
import numpy as np


def create_sample_model(path):
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10])
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 10])

    clip_node = helper.make_node(
        "Clip",
        inputs=["input"],
        outputs=["output"],
        name="legacy_clip",
        min=0.0,
        max=1.0
    )

    graph = helper.make_graph(
        [clip_node],
        "sample_graph",
        [input_tensor],
        [output_tensor]
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 11)],
        ir_version=6
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    onnx.save(model, path)
    return path
