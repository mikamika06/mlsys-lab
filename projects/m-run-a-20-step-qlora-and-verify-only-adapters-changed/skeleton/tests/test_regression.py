import sys
import numpy as np

sys.path.insert(0, ".")
from qlora.layer import LinearQLoRA
from qlora.train import train_20_steps


def test_weights_do_not_change():
    raise NotImplementedError
