import sys
sys.path.insert(0, ".")
from compilerdiag import diagnose, repro, decompositions
import ref

def test_identify_layer_valid():
    for s in ref.SCENARIOS:
        layer = diagnose.identify_layer(s["id"])
        assert layer in ["dynamo", "aot", "inductor"], f"invalid layer {layer}"
        assert layer == s["layer"], f"expected {s['layer']}, got {layer}"

def test_extract_repro_non_empty():
    for s in ref.SCENARIOS:
        code = repro.extract_repro(s["id"])
        assert isinstance(code, str) and len(code) > 0

def test_decomposition_counts_match():
    for s in ref.SCENARIOS:
        cnt = decompositions.count_decompositions(s["id"])
        assert cnt == s["decomposition_count"]
