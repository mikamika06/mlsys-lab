import torch
import torch.nn as nn
from fsdpfit.verify import verify_model_weights

def test_weights_verification():
    m1 = nn.Sequential(nn.Linear(10, 10))
    m2 = nn.Sequential(nn.Linear(10, 10))
    m2.load_state_dict(m1.state_dict())
    assert verify_model_weights(m1, m2) is True
    m3 = nn.Sequential(nn.Linear(10, 10))
    assert verify_model_weights(m1, m3) is False
