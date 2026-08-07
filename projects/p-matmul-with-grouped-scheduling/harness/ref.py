import torch


def compute_oracle(a, b):
    return torch.matmul(a, b)


def get_test_shapes():
    return [
        (128, 128, 128),
        (256, 128, 256),
        (64, 256, 128),
        (512, 512, 512),
        (111, 222, 333)
    ]
