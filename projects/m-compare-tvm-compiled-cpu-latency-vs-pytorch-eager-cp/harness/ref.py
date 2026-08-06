import time


class MockModel:

    def __init__(self, ops, complexity=1000):
        self.ops = ops
        self.complexity = complexity

    def __call__(self, x):
        acc = x
        for _ in range(self.complexity):
            acc = (acc * 1.0001 + 0.0001) % 1000.0
        return acc


SUPPORTED_OPS = {"conv2d", "relu", "add", "matmul", "batch_norm"}


def build_test_cases():
    return [
        (MockModel(["conv2d", "relu"], complexity=5000), True),
        (MockModel(["conv2d", "custom_attention"], complexity=1000), False),
        (MockModel(["matmul", "add"], complexity=5000), True),
        (MockModel(["deformable_conv"], complexity=1000), False),
    ]
