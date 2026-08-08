import numpy as np


def quantize_tensor(tensor, bits):
    raise NotImplementedError


def get_size_bytes(shape, bits):
    raise NotImplementedError


def forward(model, x):
    raise NotImplementedError


def measure_sensitivity(model, x, precisions):
    raise NotImplementedError


def build_recipe(model_shapes, sens, budget, precisions):
    raise NotImplementedError


def evaluate_recipe(model, x, recipe):
    raise NotImplementedError


def compare_recipes(model, x, budget, precisions):
    raise NotImplementedError
