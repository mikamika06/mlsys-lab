import sys
sys.path.insert(0, ".")
from actmem.formula import layer_activation_memory
from actmem.crossover import find_attention_mlp_crossover
from actmem.accounting import total_activation_memory

def test_layer_memory_scaling():
    m1 = layer_activation_memory(1, 1024, 4096, 32, 2)
    m2 = layer_activation_memory(1, 2048, 4096, 32, 2)
    assert m2 > m1 * 2

def test_crossover_validity():
    s = find_attention_mlp_crossover(1, 4096, 32, 2)
    assert s > 0

def test_total_memory_positive():
    cfg = {"layers": [{"hidden_dim": 4096, "num_heads": 32}]}
    assert total_activation_memory(cfg, 1, 1024, 2) > 0
