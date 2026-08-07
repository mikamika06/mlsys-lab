MODELS = [
    {
        "id": "model_a",
        "nodes": [
            {"name": "n1", "domain": "ai.onnx", "op_type": "MatMul", "flops": 10000},
            {"name": "n2", "domain": "custom.domain", "op_type": "CustomGeLU", "flops": 5000}
        ]
    },
    {
        "id": "model_b",
        "nodes": [
            {"name": "n1", "domain": "custom.domain", "op_type": "CustomRMSNorm", "flops": 2000}
        ]
    },
    {
        "id": "model_c",
        "nodes": [
            {"name": "n1", "domain": "ai.onnx", "op_type": "Add", "flops": 500}
        ]
    }
]

class MockV2Plugin:
    def __init__(self, fields):
        self.fields = fields
