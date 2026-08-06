OPS_15 = [
    "ADD", "SUB", "MUL", "DIV", "CONV_2D",
    "DEPTHWISE_CONV_2D", "MAX_POOL_2D", "AVERAGE_POOL_2D",
    "RESHAPE", "TRANSPOSE", "CONCATENATE", "LOGISTIC",
    "SOFTMAX", "FULLY_CONNECTED", "REDUCE_MAX"
]

def generate_test_graphs():
    graphs = []
    for i in range(5):
        ops = []
        for j in range(15):
            op_name = OPS_15[(i + j) % len(OPS_15)]
            supported = (i + j) % 2 == 0
            ops.append({
                "id": j,
                "name": op_name,
                "shape": [1, 16 + j, 16 + j, 3],
                "dtype": "FLOAT32",
                "supported": supported
            })
        graphs.append({"graph_id": i, "ops": ops})
    return graphs
