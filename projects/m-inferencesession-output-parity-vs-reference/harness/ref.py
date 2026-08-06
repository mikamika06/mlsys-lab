import numpy as np
from onnx import helper, TensorProto, checker


def get_model_bytes():
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 4])
    Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [1, 4])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 4])
    node1 = helper.make_node('Add', ['X', 'Z'], ['T1'], name='add_node')
    node2 = helper.make_node('Relu', ['T1'], ['Y'], name='relu_node')
    init_z = helper.make_tensor('Z', TensorProto.FLOAT, [1, 4], [0.0, 0.0, 0.0, 0.0])
    graph_def = helper.make_graph([node1, node2], 'test_graph', [X], [Y], initializer=[init_z])
    model_def = helper.make_model(graph_def, producer_name='mlsys-lab')
    checker.check_model(model_def)
    return model_def.SerializeToString()


def reference_inference(x_val):
    return np.maximum(0.0, x_val)
