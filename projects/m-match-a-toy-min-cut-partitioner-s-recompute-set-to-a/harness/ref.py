import torch


def get_test_module():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(16, 16)
            self.linear2 = torch.nn.Linear(16, 16)

        def forward(self, x):
            h1 = torch.relu(self.linear1(x))
            h2 = torch.relu(self.linear2(h1))
            return h2 + h1

    return ToyModel()


def compute_reference_recompute_set():
    return {"linear1", "relu1"}


def compute_reference_op_counts():
    return {"standalone_ops": 8, "compiled_ops": 5}
