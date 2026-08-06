import onnx


def build_test_model():
    node1 = onnx.helper.make_node("Identity", ["input_tensor"], ["tensor_a"])
    node2 = onnx.helper.make_node("Relu", ["tensor_a"], ["tensor_b"])
    node3 = onnx.helper.make_node("Identity", ["tensor_b"], ["output_tensor"])

    graph = onnx.helper.make_graph(
        [node1, node2, node3],
        "ref_graph",
        [onnx.helper.make_tensor_value_info("input_tensor", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("output_tensor", onnx.TensorProto.FLOAT, [1])]
    )
    return onnx.helper.make_model(graph)


def build_fuse_model():
    init_const = onnx.helper.make_tensor("const_init", onnx.TensorProto.FLOAT, [1], [0.5])
    node_mul2 = onnx.helper.make_node("Mul", ["X", "const_init"], ["mul2_out"])
    node_add = onnx.helper.make_node("Add", ["mul2_out", "const_init"], ["add_out"])
    node_erf = onnx.helper.make_node("Erf", ["add_out"], ["erf_out"])
    node_mul1 = onnx.helper.make_node("Mul", ["X", "erf_out"], ["Y"])

    graph = onnx.helper.make_graph(
        [node_mul2, node_add, node_erf, node_mul1],
        "fuse_graph",
        [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1])],
        initializer=[init_const]
    )
    return onnx.helper.make_model(graph)
