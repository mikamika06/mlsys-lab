import torch

def get_test_model():
    return torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU())

def get_test_inputs():
    return (torch.randn(4, 16),)

def get_bad_model():
    def model(x):
        if x.sum() > 0:
            return x * 2
        return x * 3
    return model
