import sys
import onnx
sys.path.insert(0, ".")
from optimizer.subgraph import extract_subgraph


def test_subgraph_extraction_bounds():
    node1 = onnx.helper.make_node("Relu", ["X"], ["Y"])
    node2 = onnx.helper.make_node("Relu", ["Y"], ["Z"])
    graph = onnx.helper.make_graph([node1, node2], "test_graph", [onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1])], [onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [1])])
    model = onnx.helper.make_model(graph)

    sub = extract_subgraph(model, ["Y"], ["Z"])
    assert len(sub.graph.node) == 1
    assert sub.graph.node[0].op_type == "Relu"
