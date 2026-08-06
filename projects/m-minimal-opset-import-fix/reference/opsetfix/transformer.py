import onnx
from onnx import helper, numpy_helper
import numpy as np


def fix_opset_and_clip(model_path, output_path, target_opset=13):
    model = onnx.load(model_path)

    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            opset.version = target_opset

    if model.ir_version < 7:
        model.ir_version = 7

    graph = model.graph
    new_nodes = []
    new_initializers = list(graph.initializer)

    for node in graph.node:
        if node.op_type == "Clip":
            inputs = list(node.input)
            attributes = {attr.name: attr for attr in node.attribute}

            min_val = None
            max_val = None

            if "min" in attributes:
                min_val = attributes["min"].f
                node.attribute.remove(attributes["min"])
            if "max" in attributes:
                max_val = attributes["max"].f
                node.attribute.remove(attributes["max"])

            while len(inputs) < 3:
                inputs.append("")

            if min_val is not None and not inputs[1]:
                init_name = f"{node.name}_min_val" if node.name else f"clip_min_{len(new_initializers)}"
                init_tensor = numpy_helper.from_array(np.array(min_val, dtype=np.float32), name=init_name)
                new_initializers.append(init_tensor)
                inputs[1] = init_name

            if max_val is not None and not inputs[2]:
                init_name = f"{node.name}_max_val" if node.name else f"clip_max_{len(new_initializers)}"
                init_tensor = numpy_helper.from_array(np.array(max_val, dtype=np.float32), name=init_name)
                new_initializers.append(init_tensor)
                inputs[2] = init_name

            new_node = helper.make_node(
                op_type="Clip",
                inputs=[inp for inp in inputs if inp != ""],
                outputs=list(node.output),
                name=node.name
            )
            new_nodes.append(new_node)
        else:
            new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)

    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    onnx.save(model, output_path)
    return output_path
