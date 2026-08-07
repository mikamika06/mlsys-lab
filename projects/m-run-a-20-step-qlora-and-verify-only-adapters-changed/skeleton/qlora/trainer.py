import numpy as np


def train_qlora_step(model_dict, x, target, lr=0.01):
    raise NotImplementedError


def run_qlora_training(model_dict, data_batches, lr=0.01):
    raise NotImplementedError


def verify_adapter_isolation(initial_base, current_model_dict):
    raise NotImplementedError
