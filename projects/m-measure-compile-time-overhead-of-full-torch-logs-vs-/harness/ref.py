TEST_CASES = [
    {"baseline_time": 10.0, "logged_time": 15.5, "expected_ratio": 1.55},
    {"baseline_time": 5.2, "logged_time": 8.0, "expected_ratio": 8.0 / 5.2},
    {"baseline_time": 20.0, "logged_time": 22.1, "expected_ratio": 22.1 / 20.0},
]

LOG_SNIPPETS = [
    ("}[TORCH_LOGS] dynamo: [WARNING] graph break due to unsupported op: aten.foo.default", "aten.foo.default"),
    ("[INFO] torch._dynamo: Step 3 failed: TorchScript lowering error at aten.bar.tensor", "aten.bar.tensor"),
    ("DEBUG:torch._inductor.lowering: unsupported node: aten.custom_op.v2 found", "aten.custom_op.v2"),
]
