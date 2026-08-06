import pytest
from actmem.formula import compute_layer_activation_bytes
from actmem.crossover import find_attention_mlp_crossover
from actmem.total import compute_total_uncheckpointed_memory

def test_formula_positive():
    val = compute_layer_activation_bytes(2, 1024, 4096, 32, 2)
    assert val > 0

def test_crossover_bounds():
    s = find_attention_mlp_crossover(4096, 32, 11008)
    assert s > 0

def test_total_memory():
    tot = compute_total_uncheckpointed_memory(32, 2, 1024, 4096, 32, 11008, 2)
    assert tot > 0
